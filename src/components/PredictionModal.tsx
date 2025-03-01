import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaBrain, FaTimes, FaUpload, FaCheck } from "react-icons/fa";

interface PredictionModalProps {
  onClose: () => void;
}

const PredictionModal: React.FC<PredictionModalProps> = ({ onClose }) => {
  const [step, setStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [selectedCondition, setSelectedCondition] = useState("");
  const [predictionResult, setPredictionResult] = useState<null | {
    risk: number;
    timeline: string;
    recommendations: string[];
  }>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map(file => file.name);
      setUploadedFiles([...uploadedFiles, ...newFiles]);
    }
  };

  const handleNextStep = () => {
    if (step === 1 && uploadedFiles.length === 0) {
      return; // Prevent proceeding without files
    }
    
    if (step === 2 && !selectedCondition) {
      return; // Prevent proceeding without selecting a condition
    }
    
    if (step === 2) {
      // Simulate processing
      setIsProcessing(true);
      setTimeout(() => {
        setIsProcessing(false);
        setPredictionResult({
          risk: 23,
          timeline: "5-10 years",
          recommendations: [
            "Increase omega-3 fatty acid intake",
            "Daily cognitive exercises",
            "Regular cardiovascular exercise",
            "Improved sleep hygiene practices"
          ]
        });
        setStep(3);
      }, 2000);
    } else {
      setStep(step + 1);
    }
  };

  const handlePrevStep = () => {
    setStep(step - 1);
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="bg-gray-800 rounded-xl p-6 w-full max-w-2xl relative"
        >
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white"
          >
            <FaTimes />
          </button>
          
          <div className="text-center mb-6">
            <FaBrain className="text-blue-400 text-4xl mx-auto mb-2" />
            <h2 className="text-2xl font-bold">NeuroX Prediction Tool</h2>
            <p className="text-gray-400">
              Analyze your brain health and receive personalized insights
            </p>
          </div>
          
          {/* Progress Steps */}
          <div className="flex items-center justify-center mb-8">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <div 
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    s === step 
                      ? "bg-blue-500 text-white" 
                      : s < step 
                        ? "bg-green-500 text-white" 
                        : "bg-gray-700 text-gray-400"
                  }`}
                >
                  {s < step ? <FaCheck /> : s}
                </div>
                {s < 3 && (
                  <div 
                    className={`h-1 w-16 ${
                      s < step ? "bg-green-500" : "bg-gray-700"
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
          
          {/* Step 1: Upload Data */}
          {step === 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h3 className="text-xl font-semibold mb-4">Upload Your Data</h3>
              <p className="text-gray-300 mb-4">
                Upload your brain scans, medical records, or genetic data for analysis.
                All data is encrypted and processed securely.
              </p>
              
              <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center mb-4">
                <FaUpload className="text-gray-400 text-3xl mx-auto mb-2" />
                <p className="text-gray-400 mb-4">Drag files here or click to browse</p>
                <input
                  type="file"
                  multiple
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="px-4 py-2 bg-blue-500 rounded-lg hover:bg-blue-600 cursor-pointer"
                >
                  Select Files
                </label>
              </div>
              
              {uploadedFiles.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-medium mb-2">Uploaded Files:</h4>
                  <ul className="bg-gray-700 rounded-lg p-2">
                    {uploadedFiles.map((file, index) => (
                      <li key={index} className="flex items-center py-1">
                        <FaCheck className="text-green-500 mr-2" />
                        {file}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
          
          {/* Step 2: Select Condition */}
          {step === 2 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h3 className="text-xl font-semibold mb-4">Select Condition of Interest</h3>
              <p className="text-gray-300 mb-4">
                Choose a neurological condition you'd like to assess risk for.
                Our AI will analyze your data to provide personalized insights.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
                {conditions.map((condition) => (
                  <div
                    key={condition}
                    onClick={() => setSelectedCondition(condition)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedCondition === condition
                        ? "bg-blue-500/30 border border-blue-500"
                        : "bg-gray-700 border border-gray-600 hover:border-blue-400"
                    }`}
                  >
                    {condition}
                  </div>
                ))}
              </div>
              
              {isProcessing && (
                <div className="text-center py-4">
                  <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p>Processing your data...</p>
                </div>
              )}
            </motion.div>
          )}
          
          {/* Step 3: Results */}
          {step === 3 && predictionResult && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h3 className="text-xl font-semibold mb-4">Your Prediction Results</h3>
              
              <div className="bg-gray-700/50 rounded-lg p-6 mb-6">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-lg">Risk Assessment:</span>
                  <span className="text-2xl font-bold text-blue-400">{predictionResult.risk}%</span>
                </div>
                
                <div className="mb-4">
                  <span className="text-lg">Potential Timeline:</span>
                  <span className="ml-2">{predictionResult.timeline}</span>
                </div>
                
                <div>
                  <h4 className="text-lg mb-2">Recommendations:</h4>
                  <ul className="bg-gray-800/50 rounded-lg p-3">
                    {predictionResult.recommendations.map((rec, index) => (
                      <li key={index} className="py-1 flex items-start">
                        <FaCheck className="text-green-500 mr-2 mt-1 flex-shrink-0" />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              <p className="text-gray-300 mb-4">
                These results are based on your uploaded data and our AI analysis. 
                For a comprehensive assessment, please consult with a healthcare professional.
              </p>
              
              <div className="text-center">
                <button
                  onClick={onClose}
                  className="px-6 py-2 bg-blue-500 rounded-lg hover:bg-blue-600"
                >
                  Close and Return to Dashboard
                </button>
              </div>
            </motion.div>
          )}
          
          {/* Navigation Buttons */}
          {step < 3 && (
            <div className="flex justify-between mt-6">
              {step > 1 ? (
                <button
                  onClick={handlePrevStep}
                  className="px-4 py-2 border border-gray-500 rounded-lg hover:bg-gray-700"
                >
                  Back
                </button>
              ) : (
                <div></div>
              )}
              
              <button
                onClick={handleNextStep}
                disabled={
                  (step === 1 && uploadedFiles.length === 0) ||
                  (step === 2 && !selectedCondition) ||
                  isProcessing
                }
                className={`px-4 py-2 rounded-lg ${
                  ((step === 1 && uploadedFiles.length === 0) ||
                  (step === 2 && !selectedCondition) ||
                  isProcessing)
                    ? "bg-blue-700/50 cursor-not-allowed"
                    : "bg-blue-500 hover:bg-blue-600"
                }`}
              >
                {isProcessing ? "Processing..." : step === 2 ? "Generate Prediction" : "Next"}
              </button>
            </div>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

const conditions = [
  "Alzheimer's Disease",
  "Parkinson's Disease",
  "Multiple Sclerosis",
  "Stroke Risk",
  "Cognitive Decline",
  "Epilepsy",
  "Migraine Patterns",
  "Memory Disorders"
];

export default PredictionModal;