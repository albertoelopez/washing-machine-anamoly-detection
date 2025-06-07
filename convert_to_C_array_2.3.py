# convert_to_c_array.py
def convert_to_c_array(tflite_path, output_path='arduino/model.h'):
    """Convert a TFLite model to a C array"""
    with open(tflite_path, 'rb') as f:
        model_content = f.read()
    
    with open(output_path, 'w') as f:
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("const unsigned char model_tflite[] = {\n")
        
        # Write 12 bytes per line
        for i in range(0, len(model_content), 12):
            chunk = model_content[i:i+12]
            f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        
        f.write("};\n")
        f.write(f"const unsigned int model_tflite_len = {len(model_content)};\n")
        f.write("\n#endif // MODEL_DATA_H\n")

if __name__ == "__main__":
    convert_to_c_array('model/model_quant.tflite')