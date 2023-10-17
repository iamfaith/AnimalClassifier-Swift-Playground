//: # Core ML demo! Use machine learning to predict animal species.
//: - Create ML: How we training our models
//: - Core ML: Use models in iOS Environment

import PlaygroundSupport

// Present the view controller in the Live View window
//PlaygroundPage.current.liveView = AnimalClassifierViewController()

//let model = AnimalClassifier()
//print(model)

// 导入Core ML框架
import CoreML
import SwiftUI

// 创建一个MLModel实例，传入你的mlmodelc文件的URL
//let modelURL = Bundle.main.url(forResource: "CourtKeypointHeatmap", withExtension: "mlmodelc")!
let modelURL = Bundle.main.url(forResource: "MakeMissMerge", withExtension: "mlmodelc")!
//let modelURL = Bundle.main.url(forResource: "MakeMissBase", withExtension: "mlmodelc")!

let model = try! MLModel(contentsOf: modelURL)


import GameplayKit
// 创建一个随机数生成器
let random = GKRandomSource()

// 创建一个空的MultiArray (Double 1280 × 1 × 3)
guard let multiArray = try? MLMultiArray(shape: [1280, 1, 3], dataType: .double) else {
    fatalError("无法创建MultiArray")
}

//guard let multiArray = try? MLMultiArray(shape: [96, 96, 3], dataType: .float32) else {
//    fatalError("无法创建MultiArray")
//}

// 遍历MultiArray的每个元素，用随机数填充
for i in 0..<multiArray.count {
    // 生成一个0到1之间的随机数
    let randomNumber = random.nextUniform()
    // 将随机数转换为NSNumber类型
    let randomValue = NSNumber(value: 0.2)
    // 将随机数赋值给MultiArray的对应元素
    multiArray[i] = randomValue
}

// Load an image from a file
//let image = UIImage(named: "image.jpg")
//// Resize the image to match the model's input shape
//let resizedImage = image?.resize(to: CGSize(width: 224, height: 224))
//// Convert the image to a pixel buffer
//let pixelBuffer = resizedImage?.pixelBuffer()


//let inputArray = MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
// Fill the array with some values
//for i in 0..<inputArray.count {
//    inputArray[i] = NSNumber(value: Float(i) / Float(inputArray.count))
//}


// Convert the array to a pixel buffer
//let pixelBuffer = try multiArray.toPixelBuffer()

// 创建一个MLFeatureValue对象，用MultiArray作为参数
let inputFeature = MLFeatureValue(multiArray: multiArray)
// 打印featureValue的描述信息
print(inputFeature.description)



let inputFeatureProvider = try! MLDictionaryFeatureProvider(dictionary: ["featureVectors" : inputFeature])
// 调用模型的prediction方法，传入输入特征提供者，得到输出特征提供者
let outputFeatureProvider = try! model.prediction(from: inputFeatureProvider)
let classProbabilities = outputFeatureProvider.featureValue(for: "classProbabilities")!.multiArrayValue!
print(classProbabilities)


//let inputFeatureProvider = try! MLDictionaryFeatureProvider(dictionary: ["image" : inputFeature])
//// 调用模型的prediction方法，传入输入特征提供者，得到输出特征提供者
//let outputFeatureProvider = try! model.prediction(from: inputFeatureProvider)
//let classProbabilities = outputFeatureProvider.featureValue(for: "featureVector")!.multiArrayValue!
//print(classProbabilities)

//print(model)

public func buffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    
    let width = 256 // Int(image.size.width)
    let height = 256 //Int(image.size.height)
//    let format = kCVPixelFormatType_24BGR // kCVPixelFormatType_32ARGB
    let format = kCVPixelFormatType_32ARGB
    
    
    var pixelBuffer : CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &pixelBuffer)
    guard (status == kCVReturnSuccess) else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
    
//    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    
    
    let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    
    context?.translateBy(x: 0, y: image.size.height)
    context?.scaleBy(x: 1.0, y: -1.0)
    
    UIGraphicsPushContext(context!)
    image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    
    return pixelBuffer
}



////// 创建一个MLFeatureProvider实例，用于存储输入图片的特征值
////// 假设你的图片是一个UIImage对象，名为image
////// 你需要将图片转换为CVPixelBuffer格式，并且调整大小为256x256
//let image = UIImage(named: "test2.png")!
//print(image)
//let pixelBuffer = buffer(from: image)
////print(pixelBuffer)
//
////let pixelBuffer = image.pixelBuffer(width: 256, height: 256)!
//let inputFeature = MLFeatureValue(pixelBuffer: pixelBuffer!)
//let inputFeatureProvider = try! MLDictionaryFeatureProvider(dictionary: ["image" : inputFeature])
//////
//////// 调用模型的prediction方法，传入输入特征提供者，得到输出特征提供者
//let outputFeatureProvider = try! model.prediction(from: inputFeatureProvider)
//////
/////
/////
/////
////// 从输出特征提供者中获取你感兴趣的特征值，例如heatmap
//let heatmap = outputFeatureProvider.featureValue(for: "heatmaps")!.multiArrayValue!
////
////// 打印或处理heatmap的值
//print(heatmap.shape)
//print("----")
//
//import CoreLocation
//
//// 假设您有一个Double 12 × 256 × 256 array，命名为doubleArray
//var coordinateArray: [CLLocationCoordinate2D] = [] // 创建一个空的CLLocationCoordinate2D数组
//for i in 0..<12 { // 遍历Double 12 × 256 × 256 array的第一维度
////    let subArray = heatmap[i]  // 获取第i个子数组
////    print(subArray)
//
//    var maxVal = 0.0
//    var currRow = 0
//    var currCol = 0
//    for row in 0..<256 {
//        for col in 0..<256 {
//            let val = heatmap[i*256*256 + row * 256 + col]
//            if Double(val) > maxVal {
//                maxVal = Double(val)
//                currRow = row
//                currCol = col
//            }
//        }
//    }
//    print(i, currRow, currCol, maxVal)
//
////    let coordinates = subArray.map { CLLocationCoordinate2D(latitude: $0[0], longitude: $0[1]) } // 将子数组中的每个元素转换为CLLocationCoordinate2D
////    coordinateArray.append(contentsOf: coordinates) // 将转换后的坐标添加到coordinateArray中
//}
