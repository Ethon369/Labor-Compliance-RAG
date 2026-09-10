#!/usr/bin/env bash
# 劳动争议智能合规助手 · 一键部署脚本
#
# 用法（在干净的云服务器上执行）：
#   bash deploy.sh
#
# 脚本做了什么：检查 Docker → 拉代码 → 生成 .env（含随机密钥）→ 启动容器 → 首次入库 → 健康检查
# 幂等设计：重复执行只会 pull 最新代码并重建，不会清掉已有数据。

set -euo pipefail

REPO="${REPO:-https://gitee.com/ren-youwen/Labor-Compliance-RAG.git}"  # 国内服务器用 gitee 更快
DIR="${DIR:-Labor-Compliance-RAG}"
COMPOSE_FILE="docker-compose.prod.yml"

log() { printf '\033[36m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; }

# ---------- 0. 小内存保护：2G 机器构建镜像时 pip 编译可能 OOM，先垫 swap ----------
MEM_KB=$(awk '/MemTotal/ {print int($2)}' /proc/meminfo)
SWAP_KB=$(awk '/SwapTotal/ {print int($2)}' /proc/meminfo)
if [ "$MEM_KB" -lt 2500000 ] && [ "$SWAP_KB" -lt 1000000 ] && [ ! -f /swapfile ]; then
  log "检测到小内存机器（$((MEM_KB/1024))MB），创建 2G swap 防构建 OOM..."
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
elif [ "$SWAP_KB" -ge 1000000 ]; then
  log "swap 已就绪（$((SWAP_KB/1024))MB）"
fi

# ---------- 1. 检查 Docker ----------
if ! command -v docker >/dev/null 2>&1; then
  log "未检测到 Docker，使用阿里云镜像源安装（官方源 download.docker.com 国内被墙）..."
  apt-get update -qq && apt-get install -y -qq ca-certificates curl >/dev/null
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
else
  log "Docker 已存在：$(docker --version | head -1)"
fi

# ---------- 1.5 配置镜像加速（国内拉 Docker Hub 镜像必备，否则 build 时拉基础镜像会卡死） ----------
if [ ! -f /etc/docker/daemon.json ] || ! grep -q registry-mirrors /etc/docker/daemon.json 2>/dev/null; then
  log "配置 Docker Hub 镜像加速..."
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<'MIRROR_EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://dockerproxy.net"
  ]
}
MIRROR_EOF
  systemctl restart docker
else
  log "镜像加速已配置"
fi

docker compose version >/dev/null 2>&1 || { err "Docker Compose 插件不可用，请升级 Docker"; exit 1; }

# ---------- 2. 获取代码 ----------
# 已经站在项目目录里就直接 pull，避免再 clone 一次形成嵌套目录
if [ -f docker-compose.prod.yml ] && [ -d .git ]; then
  log "已在项目目录内，拉取最新代码..."
  git pull --ff-only
elif [ -d "$DIR/.git" ]; then
  log "目录已存在，拉取最新代码..."
  git -C "$DIR" pull --ff-only
  cd "$DIR"
else
  log "克隆仓库：$REPO"
  git clone "$REPO" "$DIR"
  cd "$DIR"
fi

# ---------- 3. 生成 .env ----------
if [ ! -f .env ]; then
  if [ ! -f .env.production ]; then
    err "缺少 .env.production 模板"; exit 1
  fi
  log "生成 .env，并填充随机密钥..."
  cp .env.production .env

  # 自动生成强随机值，避免弱口令上线
  RAND_SECRET="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  RAND_ADMIN="$(openssl rand -hex 8  2>/dev/null || head -c 8  /dev/urandom | od -An -tx1 | tr -d ' \n')"
  RAND_PG="$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"

  # 用 | 作分隔符，避免随机值里的 / 等特殊字符冲突
  sed -i "s|^AUTH_SECRET=.*|AUTH_SECRET=${RAND_SECRET}|"       .env
  sed -i "s|^ADMIN_PASSWORD=.*|ADMIN_PASSWORD=${RAND_ADMIN}|"  .env
  sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${RAND_PG}|" .env

  log "已生成的管理员口令：$RAND_ADMIN  （请保存，稍后可自行修改 .env）"
else
  log ".env 已存在，跳过生成（如需重置请手动删除后重跑）"
fi

# ---------- 4. 启动容器 ----------
log "构建并启动服务..."
docker compose -f "$COMPOSE_FILE" up -d --build

log "等待数据库就绪..."
for i in $(seq 1 60); do
  if docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U labor -d labor >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# ---------- 5. 首次入库（ingest_laws 写的是 V1 的 articles 表） ----------
log "检查是否需要初始化法条数据..."
ARTICLES="$(docker compose -f "$COMPOSE_FILE" exec -T postgres \
  psql -U labor -d labor -tAc "SELECT count(*) FROM articles" 2>/dev/null || echo 0)"
if [ "${ARTICLES:-0}" -gt 0 ] 2>/dev/null; then
  log "已有 ${ARTICLES} 条法条源数据，跳过抓取入库"
else
  log "首次入库（抓取法条写入 articles 表）..."
  docker compose -f "$COMPOSE_FILE" exec -T app python scripts/ingest_laws.py
fi

# ---------- 5.5 迁移到默认知识库（关键！V2 检索只查 chunks，缺这步会检索不到任何条文） ----------
# articles 是 V1 遗留表，检索层统一走 kb→documents→chunks 三层模型。
# --force-embed 保证 embedding 与当前 EMBEDDING_MODE 一致（换提供方后必须重灌）。
log "迁移法条到默认知识库（kb→documents→chunks）并灌入向量..."
docker compose -f "$COMPOSE_FILE" exec -T app python scripts/migrate_default_kb.py --force-embed

# ---------- 6. 健康检查 ----------
log "健康检查..."
sleep 3
if curl -sf -o /dev/null http://127.0.0.1/; then
  IP="$(curl -s --max-time 3 ifconfig.me || echo '<服务器公网IP>')"
  log "✅ 部署完成！访问： http://${IP}"
  log "   管理员账号见上方生成记录（或 .env 里的 ADMIN_USERNAME / ADMIN_PASSWORD）"
else
  err "服务未就绪，查看日志：docker compose -f $COMPOSE_FILE logs -f"
  exit 1
fi
