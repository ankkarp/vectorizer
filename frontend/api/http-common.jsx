import axios from "axios";

export default axios.create({
  baseURL: process.env.NEXT_PUBLIC_SERVER_ADDRESS,
  headers: {
    "Content-type": "application/json",
  },
});