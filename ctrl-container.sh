# コンテナ制御スクリプト

# コンテナイメージの全削除
purge(){
  docker ps -qa|xargs docker rm -f
  docker images -qa|xargs docker rmi -f
  echo "--- docker images ---"
  docker images
  echo "--- docker processes ---"
  docker ps
}

# コンテナの起動
start() {
  docker compose up -d
}

# コンテナ名を指定して停止
stop_container_byname() {
  container_name=$1
  container_id=`docker ps -a|grep ${container_name}|cut -d" " -f1`
  echo "container_name:${container_name} container_id:${container_id}"
  docker stop ${container_id} && docker rm ${container_id}
}

# DBダンプの取得 ./mysql/sql/2_dumpdb.sql
db_dump() {
  docker compose exec mysqldb bash -c 'mysqldump --user=root --password=passw0rd db_kanpo > /docker-entrypoint-initdb.d/2_dumpdb.sql'
}

# コンテナの停止
stop() {
  # DBダンプの取得
  db_dump

  stop_container_byname streamlit
  stop_container_byname mysql_container

  status
}

# コンテナ、コンテナイメージの状態確認
status() {
  echo "--- docker ps -a ---"
  docker ps -a
  echo "--- docker images -a ---"
  docker images -a
}

# streamlit コンテナにログイン
login_web() {
  docker compose exec web bash
}

# MySQLコンソールにログイン
login_db_console() {
  docker compose exec mysqldb bash -c 'mysql --user=root --password=passw0rd '
}

# MySQLコンテナにログイン
login_db() {
  docker compose exec mysqldb bash
}

# phpmyadmin の設定初期化
db_cleanup() {
  sudo rm -rf ./mysql/data/*
}

usage() {
cat <<EOUSAGE
-----------------------------------------------------------------
Usage: $0 [command] [arg1] [arg2]....

command:  [purge|start|stop|status|login_web|login_db|login_db_console|login_phpmyadmin]

command detail:
 purge          : 全てのコンテナイメージを削除します
 start          : コンテナの起動
 stop           : コンテナの停止 / 停止前にDBバックアップを行います
 status         : コンテナの状態確認
 db_dump        : DBのバックアップ/ mysql/sql/2_dumpdb.sql としてバックアップ
 login_web        : streamlit コンテナへのログイン
 login_db         : MySQL コンテナへのログイン
 login_db_console : MySQL コンソールへのログイン
------------------------------------------------------------------
EOUSAGE
}

case $1 in
purge)
  purge
  db_cleanup
  ;;
start)
  start
  ;;
stop)
  stop
  ;;
status)
  status
  ;;
db_dump)
  db_dump
  ;;
login_web)
  login_web
  ;;
login_db)
  login_db
  ;;
login_db_console)
  login_db_console
  ;;
*)
  usage
  ;;
esac
exit 0
