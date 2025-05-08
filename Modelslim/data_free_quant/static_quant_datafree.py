#data-free 静态量化，以Resnet模型为例
# 导入squant_ptq量化接口
from modelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig  

# 可选，导入日志配置接口
from modelslim import set_logger_level  

# 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息
set_logger_level("info")  

# 使用QuantConfig接口，配置量化参数，并返回量化配置实例，当前示例使用默认配置
config = QuantConfig()   

# 配置待量化模型的输入路径，请根据实际路径配置
input_model_path = "resnet50.onnx"  

# 配置量化后模型的名称及输出路径，请根据实际路径配置
output_model_path = "resnet50_quant.onnx"  

# 使用OnnxCalibrator接口，输入待量化模型路径，量化配置数据，生成calib量化任务实例，其中calib_data为可选配置，可参考精度保持策略的方式三输入真实的数据
calib = OnnxCalibrator(input_model_path, config)   

# 执行量化
calib.run()   

# 导出量化后模型
calib.export_quant_onnx(output_model_path)  

