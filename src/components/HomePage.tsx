import React from "react";
import { motion } from "framer-motion";
import { FaBrain, FaChartLine, FaCogs, FaBookOpen, FaUserShield, FaHospitalUser, FaMicroscope, FaLightbulb, FaDatabase, FaRobot, FaStethoscope } from "react-icons/fa";

interface HomePageProps {
  onLoginClick: () => void;
  onPredictionClick: () => void;
  onLearnMoreClick: () => void;
}

const HomePage: React.FC<HomePageProps> = ({ onLoginClick, onPredictionClick, onLearnMoreClick }) => {
  return (
    <div className="bg-gray-900 text-white min-h-screen">
      {/* Navbar */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="flex justify-between items-center py-4 px-6 bg-gray-800 shadow-lg sticky top-0 z-10"
      >
        <h1 className="text-2xl font-bold flex items-center">
          <FaBrain className="mr-2 text-blue-400" /> NeuroX
        </h1>
        <a href="https://neuroxlogin.netlify.app/" className="text-white">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          Login/Signup
        </motion.button>
      </a>
      </motion.nav>
      
      {/* Hero Section */}
      <section className="text-center py-20 px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            NeuroX - AI-Powered Brain Health Companion
          </h1>
          <p className="text-lg mb-8 max-w-3xl mx-auto text-gray-300">
            Predict, analyze, and optimize neurological health using cutting-edge Generative AI technology.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onPredictionClick}
              className="px-6 py-3 bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors shadow-lg"
            >
              Try a Prediction
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLearnMoreClick}
              className="px-6 py-3 border border-white rounded-lg hover:bg-white hover:text-black transition-colors shadow-lg"
            >
              Learn More
            </motion.button>
          </div>
        </motion.div>
      </section>
      
      {/* Key Features Section */}
      <section className="py-16 px-6">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl font-semibold text-center mb-8">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {features.map((feature, index) => (
              <motion.a
              key={index}
              href={feature.link}
              target="_blank" // Opens link in a new tab
              rel="noopener noreferrer" // Security feature for links opening in a new tab
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ y: -5, boxShadow: "0 10px 25px -5px rgba(59, 130, 246, 0.5)" }}
              className="p-6 backdrop-blur-md bg-white/10 rounded-xl border border-gray-700 shadow-lg transition-all"
            >
              <feature.icon className="text-blue-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-300">{feature.description}</p>
            </motion.a>
            ))}
          </div>
        </motion.div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-6 bg-gray-800/50">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
          className="max-w-7xl mx-auto"
        >
          <h2 className="text-3xl font-semibold text-center mb-8">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {steps.map((step, index) => (
              <motion.div 
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                viewport={{ once: true }}
                className="p-6 backdrop-blur-md bg-white/10 rounded-xl border border-gray-700 shadow-lg text-center relative"
              >
                <step.icon className="text-green-400 text-4xl mb-4 mx-auto" />
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-300">{step.description}</p>
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 text-gray-300 text-2xl">→</div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Who Can Benefit? */}
      <section className="py-16 px-6">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
          className="max-w-7xl mx-auto"
        >
          <h2 className="text-3xl font-semibold text-center mb-8">Who Can Benefit?</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {beneficiaries.map((beneficiary, index) => (
              <motion.div 
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05 }}
                className="p-6 backdrop-blur-md bg-white/10 rounded-xl border border-gray-700 shadow-lg text-center"
              >
                <beneficiary.icon className="text-yellow-400 text-4xl mb-4 mx-auto" />
                <h3 className="text-xl font-semibold mb-2">{beneficiary.title}</h3>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Uses of AI */}
      <section className="py-16 px-6 bg-gray-800/50">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
          className="max-w-3xl mx-auto"
        >
          <h2 className="text-3xl font-semibold text-center mb-8">Uses of AI in Prediction & Treatment Optimization</h2>
          <p className="text-center text-lg text-gray-300">
            AI enhances accuracy in neurological disease prediction and optimizes treatments through continuous learning models, 
            data-driven diagnostics, and personalized therapy plans.
          </p>
          <div className="mt-8 text-center">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onPredictionClick}
              className="px-6 py-3 bg-blue-500 rounded-lg hover:bg-blue-600 transition-colors shadow-lg"
            >
              Start Your Brain Health Journey
            </motion.button>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 py-6 text-center text-gray-400">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center mb-6">
            <div className="flex items-center mb-4 md:mb-0">
              <FaBrain className="mr-2 text-blue-400" />
              <span className="text-xl font-bold text-white">NeuroX</span>
            </div>
            <div className="flex space-x-4">
              <a href="#" className="hover:text-blue-400 transition-colors">About</a>
              <a href="#" className="hover:text-blue-400 transition-colors">Privacy</a>
              <a href="#" className="hover:text-blue-400 transition-colors">Terms</a>
              <a href="#" className="hover:text-blue-400 transition-colors">Contact</a>
            </div>
          </div>
          <p>© 2025 NeuroX. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

const features = [
  { 
    title: "Multi-Timeline Simulation", 
    description: "Predict multiple neurological futures based on different treatment paths and lifestyle choices.", 
    icon: FaChartLine,
    link: "https://braintwin.netlify.app/" 
  },
  { 
    title: "Cognitive Twin AI", 
    description: "Track brain evolution over time with a digital twin that simulates your unique neural patterns.", 
    icon: FaBrain,
    link: "https://predictivecognitivetwin.netlify.app/" 
  },
  { 
    title: "Virtual Treatment Simulator", 
    description: "Optimize therapy strategies with AI that predicts treatment outcomes before implementation.", 
    icon: FaCogs,
    link: "https://virtualsim.netlify.app/" 
  },
  { 
    title: "MindVault AI", 
    description: "Preserve & retrieve personal memories with secure neural pattern storage technology.", 
    icon: FaBookOpen,
    link: "https://mindvaultx.netlify.app/" 
  },

];

const steps = [
  { title: "Data Collection", description: "Integrate patient data & scans from various sources into our secure platform.", icon: FaDatabase },
  { title: "AI Processing", description: "Analyze patterns & predict outcomes using our advanced neural network algorithms.", icon: FaRobot },
  { title: "Digital Twin Simulation", description: "Create a virtual brain model that mirrors your unique neurological characteristics.", icon: FaBrain },
  { title: "Clinical Insights", description: "Get AI-generated treatment strategies and personalized health recommendations.", icon: FaStethoscope }
];

const beneficiaries = [
  { title: "Neurosurgeons & Clinicians", icon: FaUserShield },
  { title: "Neurological Researchers", icon: FaMicroscope },
  { title: "Hospitals & Institutions", icon: FaHospitalUser },
  { title: "MedTech & AI Enthusiasts", icon: FaLightbulb }
];

export default HomePage;