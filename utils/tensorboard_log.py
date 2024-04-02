from tensorboard.backend.event_processing import event_accumulator

# 定义一个函数来转换日志文件
def convert_tfevents_to_txt(input_file, output_file):
    # 创建一个EventAccumulator对象
    ea = event_accumulator.EventAccumulator(input_file)

    # 重新加载数据
    ea.Reload()

    # 获取所有标量事件的键
    scalar_keys = ea.scalars.Keys()

    # 打开输出文件
    with open(output_file, 'w') as f:
        # 写入标量事件的键
        f.write("Scalar Keys:\n")
        f.write(", ".join(scalar_keys) + "\n\n")

        # 遍历每个标量键并写入其事件
        for key in scalar_keys:
            events = ea.scalars.Items(key)
            step_values = [(e.step, e.value) for e in events]
            f.write(f"{key}\n")
            f.write(str(step_values) + "\n\n")

# 调用函数，传入输入文件和输出文件的路径
convert_tfevents_to_txt('E:\events.out.tfevents.1711525980.gpu1.11604.0', 'output.txt')