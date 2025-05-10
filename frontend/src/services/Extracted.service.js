import axios from "axios";

const API_URL = "http://127.0.0.1:8000/api"; // chỉnh lại nếu backend chạy port khác

const ExtractedService = {
  sendPdf: async ({ pdf }) => {
    const formData = new FormData();
    formData.append("pdf", pdf);

    try {
      const response = await axios.post(`${API_URL}/extract-images/`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log(response)
      return response.data.images;
    } catch (error) {
      console.error("Lỗi khi gửi PDF:", error);
      throw error;
    }
  },

  classify: async(imageList, selectedModel, threshold) => {
    const formData = new FormData();
  
    formData.append('model', selectedModel);
    formData.append('images', JSON.stringify(imageList));
    formData.append('threshold', threshold)
    console.log(imageList)
    try {
      const response = await axios.post(`${API_URL}/classify-images/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data', 
        },
      });
      console.log(response)
      return response.data; 
    } catch (error) {
      console.error('Error during classification:', error);
      return { error: 'Error during classification' };
    }      
  }
};

export default ExtractedService;
