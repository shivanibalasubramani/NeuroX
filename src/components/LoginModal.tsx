import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaBrain, FaTimes } from "react-icons/fa";

interface LoginModalProps {
  onClose: () => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ onClose }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setMessage(isLogin ? "Login successful!" : "Account created successfully!");
      
      // Close modal after success message
      setTimeout(() => {
        onClose();
      }, 1500);
    }, 1000);
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="bg-gray-800 rounded-xl p-6 w-full max-w-md relative"
        >
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white"
          >
            <FaTimes />
          </button>
          
          <div className="text-center mb-6">
            <FaBrain className="text-blue-400 text-4xl mx-auto mb-2" />
            <h2 className="text-2xl font-bold">{isLogin ? "Welcome Back" : "Join NeuroX"}</h2>
            <p className="text-gray-400">
              {isLogin 
                ? "Access your brain health dashboard" 
                : "Start your journey to optimal brain health"}
            </p>
          </div>
          
          {message && (
            <div className="mb-4 p-3 bg-green-500/20 border border-green-500 rounded-lg text-center">
              {message}
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            {!isLogin && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-1">Full Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full p-2 rounded-lg bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>
            )}
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-2 rounded-lg bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                required
              />
            </div>
            
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full p-2 rounded-lg bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                required
              />
            </div>
            
            <button
              type="submit"
              disabled={isSubmitting}
              className={`w-full py-2 rounded-lg font-medium ${
                isSubmitting 
                  ? "bg-blue-700 cursor-not-allowed" 
                  : "bg-blue-500 hover:bg-blue-600"
              }`}
            >
              {isSubmitting 
                ? "Processing..." 
                : isLogin ? "Login" : "Create Account"}
            </button>
          </form>
          
          <div className="mt-4 text-center">
            <button 
              onClick={() => setIsLogin(!isLogin)}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              {isLogin 
                ? "Don't have an account? Sign up" 
                : "Already have an account? Login"}
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default LoginModal;