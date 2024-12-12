import tensorflow as tf
import numpy as np
import os

# 设置模型文件路径
checkpoint_index_path = '/root/workspace/EasyRec/examples/ckpt/deepfm_movieslen_ckpt/model.ckpt-650.index'
checkpoint_Meta_path = '/root/workspace/EasyRec/examples/ckpt/deepfm_movieslen_ckpt/model.ckpt-650.meta'

# 加载模型图结构
def load_model(checkpoint_file):
    # 使用tf.train.import_meta_graph加载图结构
    tf.compat.v1.disable_eager_execution()  # 禁用eager execution以便使用旧API
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(checkpoint_Meta_path)
        return graph, saver

# 加载模型
graph, saver = load_model(checkpoint_Meta_path)

# 解析一条数据
data = "1272 2395 5 1 4 20 49506 1 1 1"
fields = data.split()
print(f"fields: {fields}")
numeric_fields = [int(field) if field.isdigit() else field for field in fields]
numeric_data = [field for field in numeric_fields if isinstance(field, int)]
print(f"numeric_data: {numeric_data}")

# 创建会话并加载模型参数
with tf.compat.v1.Session(graph=graph) as sess:
    try:
        # 尝试恢复模型参数
        saver.restore(sess, checkpoint_index_path)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

        # 如果检查点文件缺少某些变量，尝试手动创建并初始化它们
        try:
            # 创建并初始化 beta1_power 变量
            beta1_power = tf.compat.v1.get_variable('beta1_power', initializer=0.9)
            sess.run(beta1_power.initializer)
            
            # 再次尝试恢复模型参数
            saver.restore(sess, checkpoint_index_path)
        except Exception as e:
            print(f"手动创建变量时发生错误: {e}")
            raise

    # 获取输入和输出节点
    input_tensor = graph.get_tensor_by_name('input_tensor_name:0')  # 替换为实际的输入节点名称
    output_tensor = graph.get_tensor_by_name('output_tensor_name:0')  # 替换为实际的输出节点名称
    
    # 将数据转换为NumPy数组
    test_data = np.array([numeric_data], dtype=np.float32)
    
    # 进行预测
    prediction = sess.run(output_tensor, feed_dict={input_tensor: test_data})
    
    # 打印预测结果
    print("Prediction:", prediction)