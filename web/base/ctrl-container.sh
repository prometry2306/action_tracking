#!/bin/bash
# コンテナ操作スクリプト
#
# コマンド一覧
# purge                             : 全てのコンテナイメージを削除します
# lint                              : ruff による静的チェックを行います
# build                             : build docker image
# start                             : start docker container
# stop                              : stop docker container
# status                            : view list docker container list
# restart                           : restart docker container
# login                             : bash login to local docker container
# push [NAME_ENV]                   : docker image push to ecr repository and tag by NAME_ENV(etc. stg prod)
# ecr_tagging [TARGET_TAG] [TO_TAG] : change ECR Tag(TARGET_TAG) to TO_TAG
# set_prod_ecrtag                   : ECRのprodタグ を stgタグに移動します。
# ecs_dryrun [NAME_ENV]             : test ecs task definision json file
# ecs [NAME_ENV]                    : upload and regist ecs task definision json file
export AWS_DEFAULT_REGION=us-east-1

# 変数のセット
setting() {
  if [ -z "$1" ]; then #もし引数に値が入ってなければ
    NAME_ENV=stg
  else
    NAME_ENV=$1
  fi
  echo target env: $NAME_ENV 
  IMAGE_NAME=mmpose-base
  LATEST=latest
  FILE_ENV=../../env-file/${NAME_ENV}.env
}

# コマンドチェック
check_command() {
  which jq
  if [ $? != 0 ]; then
    echo "command jq is not installed"
    exit -1
  fi
  which ruff
  if [ $? != 0 ]; then
    echo "command ruff is not installed"
    echo "pip3 install ruff"
    exit 0
  fi

}

# AWSアカウントチェック
check_AWSID() {
  check_command
  AWS_AccountID=`aws sts get-caller-identity|jq .Arn|sed -e 's/"//g'|cut -d ":" -f5`
  echo "AWS_AccountID=${AWS_AccountID}"
  export AWS_AccountID
  DOMAIN=public.ecr.aws/d4g9d0l2
  export DOMAIN
}

# コンテナイメージの全削除
purge(){
  docker ps -qa|xargs docker rm -f
  docker images -qa|xargs docker rmi -f
  docker builder prune -af
  echo "--- docker images ---"
  docker images
  echo "--- docker processes ---"
  docker ps
}

# 静的チェック
lint() {
  result=$(ruff . 2>&1)
  if [[ $result ]]; then
    echo "ERROR : lint error detected."
    echo $result
    exit 0
  else
    exit 0
  fi
}

# コンテナのビルド
build() {
  if [ -f init.sh ]; then
    bash init.sh
  fi
  docker build -t ${IMAGE_NAME} .
}

# コンテナの起動
start() {
  echo "--- starting container  ---"
  DIR_LOCAL=`pwd`/app/
  DIR_CONTAINER=/var/batch/
  mkdir -p app/logs/ app/uploaded-files/
  chmod 777 -R app/logs/ app/uploaded-files/

  container_id=`docker ps -a |grep ${IMAGE_NAME}|cut -d" " -f1`
  if [ "${container_id}" != "" ]; then 
    docker rm -f ${container_id}
  fi
  GPU_CONF=""
  #GPU_CONF="--gpus all"

  image_id=`docker images|grep ^${IMAGE_NAME}| sed -e 's/  */ /g'|cut -d" " -f3`
  #echo docker run -d --env-file env.txt --name ${IMAGE_NAME} ${image_id}
  echo docker run -p 80:80 ${GPU_CONF} --env-file ${FILE_ENV} -v ${DIR_LOCAL}:${DIR_CONTAINER} --name ${IMAGE_NAME} -itd ${image_id}
  docker run -p 80:80 ${GPU_CONF} --env-file ${FILE_ENV} -v ${DIR_LOCAL}:${DIR_CONTAINER} --name ${IMAGE_NAME} -itd ${image_id}
  echo ""
  docker logs ${IMAGE_NAME}
}

# コンテナの停止
stop() {
  echo "--- stopping container  ---"
  container_id=`docker ps -a |grep ${IMAGE_NAME}|cut -d" " -f1`
  echo $container_id
  if [ "${container_id}" != "" ]; then
    docker stop ${container_id} && docker rm -f ${container_id}
  fi
  echo ""
}

# コンテナの状態確認
status() {
  echo "--- check container status  ---"
  docker ps
  echo ""
}

# コンテナの中に入る
login() {
  echo "--- start container  ---"
  container_id=`docker ps -a |grep -v CONTAINER |grep "${IMAGE_NAME}"|cut -d" " -f1`
  echo "container:${container_id}"
  if [ "${container_id}" == "" ]; then
    start
    sleep 1
    container_id=`docker ps|grep -v IMAGE|cut -d " " -f1`
  fi
  docker exec -it ${container_id} bash
}

# コンテナをECRにプッシュ
push() {
  DOMAIN=public.ecr.aws/d4g9d0l2
  URL_REPO=${DOMAIN}/${IMAGE_NAME}

  # 1. login
  #echo aws ecr get-login-password | docker login --username AWS --password-stdin ${DOMAIN}
  #aws ecr get-login-password | docker login --username AWS --password-stdin ${DOMAIN}
  echo aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/d4g9d0l2
  aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/d4g9d0l2

  # 2. tagging
  #echo docker tag ${IMAGE_NAME}:${LATEST} ${URL_REPO}:${LATEST}
  #docker tag ${IMAGE_NAME}:${LATEST} ${URL_REPO}:${LATEST}
  echo docker tag mmpose-base:latest public.ecr.aws/d4g9d0l2/ubuntu22-cuda11.8-mmpose:latest
  docker tag mmpose-base:latest public.ecr.aws/d4g9d0l2/ubuntu22-cuda11.8-mmpose:latest

  # 3. register to ECR
  #echo docker push ${URL_REPO}:${LATEST}
  #docker push ${URL_REPO}:${LATEST}
  echo docker push public.ecr.aws/d4g9d0l2/ubuntu22-cuda11.8-mmpose:latest
  docker push public.ecr.aws/d4g9d0l2/ubuntu22-cuda11.8-mmpose:latest
}

ecr_tagging() {
  FROM_DIGEST=$(aws ecr list-images --repository-name ${IMAGE_NAME} --query "imageIds[?imageTag=='$1'] | [0].imageDigest")
  TO_DIGEST=$(aws ecr list-images --repository-name ${IMAGE_NAME} --query "imageIds[?imageTag=='$2'] | [0].imageDigest")
  if [[ "$FROM_DIGEST" = "$TO_DIGEST" ]]
  then
    # 既にコピー元とコピー先が同じダイジェストの場合は何もしない
    echo "Tag already same. Skipped."
  else
    # リモートでタグをコピー
    echo "ecr tag copy: ${1} => ${2}"
    # https://docs.aws.amazon.com/cli/latest/reference/ecr/batch-get-image.html
    MANIFEST=$(aws ecr batch-get-image --repository-name ${IMAGE_NAME} --image-ids imageTag=$2 --query 'images[].imageManifest' --output text)

    # https://docs.aws.amazon.com/cli/latest/reference/ecr/put-image.html
    aws ecr put-image --repository-name ${IMAGE_NAME} --image-tag $1 --image-manifest "$MANIFEST"
  fi
}

# ECRのprodタグをstgタグの位置に揃える
set_prod_ecrtag() {
  check_AWSID
  # 1. login
  aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin ${DOMAIN}

  ecr_tagging prod_bak prod
  ecr_tagging prod stg
}

# ECS task の登録のドライラン
register_ecs_task_dryrun() {
  aws ecs register-task-definition --family fargate-efs-mount-test --cli-input-json file://${FILE_ECS}
}

# ECS task の登録
register_ecs_task() {
  aws ecs register-task-definition --cli-input-json file://${FILE_ECS}
}

usage() {
cat <<EOUSAGE
-----------------------------------------------------------------
Usage: $0 [command] [arg1] [arg2]....

command:  [start|stop|restart|status|build|push|ecs_dryrun|ecs|login|ecr_tagging]

command detail:
purge                             : remove all docker images
lint                              : ruff による静的チェックを行います
build                             : build docker image
start [NAME_ENV]                  : start docker container
stop                              : stop docker container
status                            : view list docker container list
restart [NAME_ENV]                : restart docker container
login [NAME_ENV]                  : bash login to local docker container
push [NAME_ENV]                   : docker image push to ecr repository and tag by NAME_ENV(etc. stg prod)
ecr_tagging [TARGET_TAG] [TO_TAG] : change ECR Tag(TARGET_TAG) to TO_TAG
set_prod_ecrtag                   : ECRのprodタグ を stgタグに移動します。
ecs_dryrun [NAME_ENV]             : test ecs task definision json file
ecs [NAME_ENV]                    : upload and regist ecs task definision json file
------------------------------------------------------------------
EOUSAGE
}


case $1 in
purge)
  purge
  ;;
lint)
  lint
  ;;
build)
  setting
  build
  ;;
start)
  setting $2
  stop
  start
  status
  ;;
stop)
  setting
  stop
  status
  ;;
status)
  setting
  status
  ;;
restart)
  setting $2
  stop
  start
  status
  ;;
login)
  setting $2
  login
  ;;
push)
  setting $2
  push
  ;;
ecr_tagging)
  setting $2
  ecr_tagging $2 $3
  ;;
set_prod_ecrtag)
  setting
  set_prod_ecrtag $2 $3
  ;;
ecs_dryrun)
  setting $2
  register_ecs_task_dryrun
  ;;
ecs)
  setting $2
  register_ecs_task
  ;;
*)
  usage
  ;;
esac
exit 0
