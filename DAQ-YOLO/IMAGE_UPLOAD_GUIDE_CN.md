# GitHub 图片上传指南（约 100 张）

这个代码包已经预先创建了图片目录。建议先把代码包中的文件上传到 GitHub，再进入对应图片目录上传图片。

## 最简单的上传位置

将普通项目图片统一上传到：

```text
images/uploads/
```

如果希望分类整理，也可以使用：

```text
images/architecture/   # 网络结构图、流程图
images/results/        # 检测结果图、对比图、消融实验图
images/samples/        # 少量示例输入图片
```

## 推荐操作顺序

1. 解压本代码包。
2. 在 GitHub 新建仓库，例如 `DAQ-YOLO`。
3. 先上传解压后的全部代码文件和文件夹。
4. 打开 GitHub 仓库中的 `images/uploads/`。
5. 点击 `Add file` → `Upload files`。
6. 将图片拖入网页。
7. 约 100 张图片建议分成两批上传，例如每批 40～60 张，上传失败时更容易重试。
8. 每批完成后点击 `Commit changes`。

## 图片命名建议

推荐：

```text
seedling-001.jpg
seedling-002.jpg
result-001.png
architecture-dema.png
```

避免：

```text
新建文件夹 (1) 最终最终版图片 01.png
```

文件名尽量只使用英文、数字、短横线和下划线。

## 在 README 中显示图片

上传后可在根目录 `README.md` 中写：

```markdown
![检测结果](images/uploads/result-001.png)
```

分类目录示例：

```markdown
![DAQ-YOLO 网络结构](images/architecture/daq-yolo-architecture.png)
![检测效果](images/results/detection-result-001.jpg)
```

## 不建议上传的内容

不要上传以下内容：

- 完整的大型训练数据集；
- 含个人隐私或未授权使用的图片；
- API Key、密码、Token；
- 大型模型权重，如 `.pt`、`.onnx`、`.engine`；
- `runs/` 训练缓存目录。

如果这约 100 张图片是公开展示、论文插图或少量测试样例，可以直接放进本仓库；如果它们属于完整训练数据集的一部分，则更建议单独托管数据集。
