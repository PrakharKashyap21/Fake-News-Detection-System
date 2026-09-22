import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8008";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15000,
});

export const predictNews = async (title, text) => {
  try {
    const response = await apiClient.post("/predict", {
      title: title || "",
      text: text || "",
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      if (error.response.status === 422) {
        const detail = error.response.data?.detail;
        if (typeof detail === "string") {
          throw new Error(detail);
        } else if (Array.isArray(detail) && detail.length > 0) {
          throw new Error(detail[0].msg || "Validation error: Please enter valid headline or text.");
        }
        throw new Error("Validation error: Please enter a headline or article text.");
      }
      throw new Error(error.response.data?.detail || "Unable to process prediction request.");
    } else if (error.request) {
      throw new Error("Unable to connect to the backend server. Please check if the API is running.");
    } else {
      throw new Error("An unexpected error occurred while making the request.");
    }
  }
};

export const verifyNewsV2 = async (title, text) => {
  try {
    const response = await apiClient.post("/v2/verify", {
      title: title || "",
      text: text || "",
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      if (error.response.status === 422) {
        const detail = error.response.data?.detail;
        if (typeof detail === "string") {
          throw new Error(detail);
        } else if (Array.isArray(detail) && detail.length > 0) {
          throw new Error(detail[0].msg || "Validation error: Please enter valid title or text.");
        }
        throw new Error("Validation error: At least a title or text is required for verification.");
      }
      throw new Error(error.response.data?.detail || "Unable to complete real-time news verification.");
    } else if (error.request) {
      throw new Error("Unable to connect to the backend verification server. Please check if the API is running.");
    } else {
      throw new Error("An unexpected error occurred during verification.");
    }
  }
};

