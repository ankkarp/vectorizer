import styles from "./FileInput.module.css";

import { useState, useRef } from "react";
import UploadIcon from "../../icons/UploadIcon/UploadIcon";
import http from "../../../api/http-common";
import Router from "next/router";
import LoadingIcon from "../../icons/LoadingIcon/LoadingIcon";
import Image from "next/image";

export default function FileInput() {
  const [loading, setLoading] = useState(false);
  const [resultImage, setResultImage] = useState(null)
  const inputRef = useRef();

  const updateResults = (r) => {
    setResultImage(r.data.image)
  };


  const handleFileChange = (e) => {
    e.preventDefault();
    const file = e.target.files[0];
    let formData = new FormData();
    formData.append("image", file);
    try {
      http.post("upload", formData, {
          headers: {
            "Content-Type": file.type,
          },
        })
        .then((r) => setResultImage(r.data.image));
      setLoading(true);
    } catch (e) {
      console.log(e);
    }
  };

  const handleChoose = (e) => {
    inputRef.current.click();
  };

  return (
    <div className={styles.container}>
      <div className={styles.upload}>
        {loading ? (
          <LoadingIcon />
        ) : (
          <button onClick={handleChoose}>
            <input
              ref={inputRef}
              type="file"
              onChange={handleFileChange}
              disabled={loading}
              accept=".png,.jpeg,.jpg"
            />
            <UploadIcon width={200} height={200} />
            <div className="footer">Загрузите изображение</div>
          </button>
        )}
        {resultImage&&resultImage}
      </div>
    </div>
  );
}