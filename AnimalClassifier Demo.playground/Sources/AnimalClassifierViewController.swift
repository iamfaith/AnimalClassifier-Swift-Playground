import UIKit
import CoreML
import PlaygroundSupport

public class AnimalClassifierViewController: UIViewController {
    
    let model = AnimalClassifier()
    let imageView = UIImageView()
    public let imagePicker = UIImagePickerController()
    let hintLabel = UILabel()
    let classLabel = UILabel()
    let classLabelProbability = UILabel()
    
    public override func loadView() {
        imageView.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        imagePicker.delegate = self
        
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(imageViewTapped))
        imageView.addGestureRecognizer(tapGestureRecognizer)
        imageView.isUserInteractionEnabled = true
        imageView.contentMode = .scaleAspectFill
        
        // hintLabel
        hintLabel.text = "Tap to add animal photo to get prediction result."
        hintLabel.textColor = #colorLiteral(red: 0.6000000238, green: 0.6000000238, blue: 0.6000000238, alpha: 1)
        hintLabel.textAlignment = .center
        hintLabel.font = UIFont.systemFont(ofSize: 12)
        hintLabel.translatesAutoresizingMaskIntoConstraints = false
        imageView.addSubview(hintLabel)
        
        // blackPanel
        let blackPanel = UIView()
        blackPanel.backgroundColor = #colorLiteral(red: 0, green: 0, blue: 0, alpha: 0.5)
        blackPanel.translatesAutoresizingMaskIntoConstraints = false
        imageView.addSubview(blackPanel)
        
        // classLabel
        classLabel.text = "Class"
        classLabel.textColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        classLabel.textAlignment = .center
        classLabel.font = UIFont.systemFont(ofSize: 30)
        classLabel.translatesAutoresizingMaskIntoConstraints = false
        classLabel.isHidden = true
        imageView.addSubview(classLabel)
        
        // classLabelProbability
        classLabelProbability.text = "0.0"
        classLabelProbability.textColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        classLabelProbability.textAlignment = .center
        classLabelProbability.font = UIFont.systemFont(ofSize: 24)
        classLabelProbability.translatesAutoresizingMaskIntoConstraints = false
        classLabelProbability.isHidden = true
        imageView.addSubview(classLabelProbability)
        
        
        self.view = imageView
        
        NSLayoutConstraint.activate([
            hintLabel.widthAnchor.constraint(equalToConstant: 400),
            hintLabel.heightAnchor.constraint(equalToConstant: 60),
            hintLabel.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            hintLabel.centerYAnchor.constraint(equalTo: imageView.centerYAnchor),
            
            classLabel.widthAnchor.constraint(equalToConstant: 200),
            classLabel.heightAnchor.constraint(equalToConstant: 54),
            classLabel.centerXAnchor.constraint(equalTo: imageView.centerXAnchor, constant: -110),
            classLabel.bottomAnchor.constraint(equalTo: imageView.bottomAnchor, constant: -24),
            
            classLabelProbability.widthAnchor.constraint(equalToConstant: 200),
            classLabelProbability.heightAnchor.constraint(equalToConstant: 54),
            classLabelProbability.centerXAnchor.constraint(equalTo: imageView.centerXAnchor, constant: 110),
            classLabelProbability.bottomAnchor.constraint(equalTo: imageView.bottomAnchor, constant: -24),
            
            blackPanel.widthAnchor.constraint(equalToConstant: 1000),
            blackPanel.heightAnchor.constraint(equalToConstant: 100),
            blackPanel.centerXAnchor.constraint(equalTo: imageView.centerXAnchor),
            blackPanel.bottomAnchor.constraint(equalTo: imageView.bottomAnchor)
            ])
    }
    
    public func buffer(from image: UIImage) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: image.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
    
    public func getPredictionFromModel() {
        
        let image = imageView.image!
        
        let pixelBuffer = buffer(from: image)
        
        do {
            let animal = try model.prediction(image: pixelBuffer!)
            classLabel.text = animal.classLabel
            classLabelProbability.text = "\(animal.classLabelProbs[classLabel.text!]!)"
        } catch {
            print(error)
        }
    }
    
    @objc func imageViewTapped() {
        hintLabel.isHidden = true
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
        
        present(imagePicker, animated: true, completion: nil)
    }
    
}

extension AnimalClassifierViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.image = pickedImage
            getPredictionFromModel()
            classLabel.isHidden = false
            classLabelProbability.isHidden = false
        }
        imagePicker.dismiss(animated: true, completion: nil)
    }
    
    public func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        imagePicker.dismiss(animated: true, completion: nil)
    }
}
