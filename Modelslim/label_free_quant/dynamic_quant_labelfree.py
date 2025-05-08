#label-free模式动态模型量化，以yolo为例
# 导入post_training_quant量化接口 
from modelslim.onnx.post_training_quant import QuantConfig, run_quantize 
from modelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_coco

# 可选，导入日志配置接口
from modelslim import set_logger_level  

 #可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息
set_logger_level("info")  

# 准备一小批矫正数据集，读取数据集进行数据预处理，并将数据存入calib_data，当前配置示例为空时，将随机生成矫正数据
def custom_read_data():
    calib_data = preprocess_func_coco(height=640,width=640,data_path="coco")
    # 可读取数据集，进行数据预处理，将数据存入calib_data
    return calib_data
calib_data = custom_read_data()

# 使用QuantConfig接口，配置量化参数，返回量化配置实例，当前示例中is_dynamic_shape和input_shape参数在动态shape场景下必须配置。
quant_config = QuantConfig(is_signed_quant=False,calib_data = calib_data, amp_num = 5, is_dynamic_shape = True, input_shape = [[1,3,640,640]])  

# 配置待量化模型的输入路径，请根据实际路径配置
input_model_path = "yolov5m.onnx"  

# 配置量化后模型的名称及输出路径，请根据实际路径配置
output_model_path = "yolov5m_quant_lablefree.onnx"  

# 使用run_quantize接口执行量化，配置待量化模型和量化后模型的路径及名称
run_quantize(input_model_path,output_model_path,quant_config)  
