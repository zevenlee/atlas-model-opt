#label-free模式静态模型量化，以resnet模型为例
# 导入post_training_quant量化接口
from modelslim.onnx.post_training_quant import QuantConfig, run_quantize  

# 导入预置的ImageNet数据集预处理函数preprocess_func_imagenet
from modelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet  

# 可选，导入日志配置接口
from modelslim import set_logger_level  

# 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息
set_logger_level("info")  

# 准备一小批矫正数据集，读取数据集进行数据预处理，并将数据存入calib_data
def custom_read_data():
    calib_data = preprocess_func_imagenet("images")  
    # 调用数据集预处理函数，请根据数据集实际路径配置，不使用该预处理函数时请参考数据预处理自行配置
    return calib_data
calib_data = custom_read_data()

# 使用QuantConfig接口，配置量化参数，返回量化配置实例
quant_config = QuantConfig(is_signed_quant=False,calib_data = calib_data, amp_num = 5)  

# 配置待量化模型的输入路径，请根据实际路径配置
input_model_path = "resnet101_v11.onnx"  

# 配置量化后模型的名称及输出路径，请根据实际路径配置
output_model_path = "resnet101_quant_labelfree.onnx"  

# 使用run_quantize接口执行量化，配置待量化模型和量化后模型的路径及名称，
run_quantize(input_model_path,output_model_path,quant_config)  
