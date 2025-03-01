import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaBrain, FaTimes, FaChartLine, FaUserMd, FaShieldAlt, FaDatabase } from "react-icons/fa";

interface InfoModalProps {
  onClose: () => void;
}

const InfoModal: React.FC<InfoModalProps> = ({ onClose }) => {
  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="bg-gray-800 rounded-xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto relative"
        >
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white"
          >
            <FaTimes />
          </button>
          
          <div className="text-center mb-6">
            <FaBrain className="text-blue-400 text-4xl mx-auto mb-2" />
            <h2 className="text-2xl font-bold">About NeuroX</h2>
            <p className="text-gray-400">
              The future of neurological health prediction and optimization
            </p>
          </div>
          
          <div className="space-y-6">
            <section>
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <FaChartLine className="text-blue-400 mr-2" />
                Our Technology
              </h3>
              <p className="text-gray-300">
                NeuroX uses cutting-edge artificial intelligence and machine learning algorithms to analyze neurological data and predict potential health outcomes. Our technology creates a digital twin of your brain, allowing for personalized simulations and predictions that can help optimize treatment strategies and improve overall brain health.
              </p>
            </section>
            
            <section>
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <FaUserMd className="text-blue-400 mr-2" />
                Clinical Applications
              </h3>
              <p className="text-gray-300">
                Healthcare providers can use NeuroX to:
              </p>
              <ul className="list-disc pl-6 mt-2 text-gray-300 space-y-1">
                <li>Predict disease progression with greater accuracy</li>
                <li>Simulate treatment outcomes before implementation</li>
                <li>Develop personalized care plans based on individual neural profiles</li>
                <li>Monitor changes in brain health over time</li>
                <li>Identify early warning signs of neurological conditions</li>
              </ul>
            </section>
            
            <section>
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <FaShieldAlt className="text-blue-400 mr-2" />
                Privacy & Security
              </h3>
              <p className="text-gray-300">
                We understand the sensitivity of neurological data. NeuroX employs state-of-the-art encryption and security protocols to ensure your information remains private and protected. Our platform is compliant with all relevant healthcare data regulations, and you maintain complete control over your data at all times.
              </p>
            </section>
            
            <section>
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <FaDatabase className="text-blue-400 mr-2" />
                How We Use Your Data
              </h3>
              <p className="text-gray-300">
                When you upload neurological scans, medical records, or other health data to NeuroX, our AI analyzes this information to create your personalized brain model. This model enables us to:
              </p>
              <ul className="list-disc pl-6 mt-2 text-gray-300 space-y-1">
                <li>Generate accurate predictions about potential neurological conditions</li>
                <li>Simulate how different treatments might affect your brain</li>
                <li>Provide personalized recommendations for brain health optimization</li>
                <li>Track changes in your neurological health over time</li>
              </ul>
              <p className="text-gray-300 mt-2">
                Your data is never sold to third parties and is only used for the purposes you explicitly consent to.
              </p>
            </section>
            
            <div className="mt-8 text-center">
              <p className="text-gray-300 mb-4">
                Ready to experience the future of neurological health?
              </p>
              <button
                onClick={onClose}
                className="px-6 py-2 bg-blue-500 rounded-lg hover:bg-blue-600"
              >
                Return to NeuroX
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default InfoModal;