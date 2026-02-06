# 简介
slam

# 构建
```
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/Users/qinziwen/Tools/vcpkg/scripts/buildsystems/vcpkg.cmake
make -j8
```

# 运行
```
# 直接使用命令行参数
./apps/main --dataset_dir=../dataset/tum_rgbd --sequence=rgbd_dataset_freiburg1_desk

# 使用配置文件（仅覆盖默认值）
./apps/main --config=../config/default.cfg
```

# 配置说明
配置文件为 `key=value` 格式，key 与命令行 flag 同名，命令行参数优先生效。
可用配置项示例见 `config/default.cfg`。
