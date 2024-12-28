#!/bin/sh

# 设置固定的路径来存储数据
MOBILE_MANIPULATION_CENTRAL_BAG_DIR=~/home/d/workspace/rosbags  # 固定路径

# 创建以当前日期命名的子目录，用于存储记录的数据
BAG_DIR=$MOBILE_MANIPULATION_CENTRAL_BAG_DIR/$(date +"%Y-%m-%d")

# 创建目录，-p表示如果上级目录不存在则创建
mkdir -p "$BAG_DIR"

# 使用rosbag命令记录数据
# 1. 指定输出目录和文件名（文件名前缀为传递给脚本的第一个参数 $1）
# 2. 记录 /clock 话题
# 3. 使用正则表达式匹配其他多个话题：
#    - /ridgeback/(.*)：匹配与 ridgeback 相关的所有话题
#    - /ridgeback_velocity_controller/(.*)：匹配与 ridgeback_velocity_controller 相关的所有话题
#    - /ur10/(.*)：匹配与 ur10 相关的所有话题
#    - /vicon/(.*)：匹配与 vicon 相关的所有话题
rosbag record -o "$BAG_DIR/$1" \
  /clock \
  --regex "/ridgeback/(.*)" \
  --regex "/ridgeback_velocity_controller/(.*)" \
  --regex "/ur10/(.*)" \
  --regex "/vicon/(.*)"
