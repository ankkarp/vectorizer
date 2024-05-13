import styles from "./FileInput.module.css";

import { useState, useRef } from "react";
import UploadIcon from "../../icons/UploadIcon/UploadIcon";
import http from "../../../api/http-common";
import Router from "next/router";
import LoadingIcon from "../../icons/LoadingIcon/LoadingIcon";
import Image from "next/image";

export default function FileInput({ setSvgCode, setProgressGif }) {
  const [loading, setLoading] = useState(false);
  const [inputImageURL, setInputImageURL] = useState(null);
  const inputRef = useRef();

  const updateResults = (svgCode) => {
    setSvgCode(svgCode);
    setLoading(false);
  };

  const handleFileChange = (e) => {
    e.preventDefault();
    const file = e.target.files[0];
    setInputImageURL(URL.createObjectURL(file));
    let formData = new FormData();
    formData.append("image", file);
    try {
      http
        .post("upload", formData, {
          headers: {
            "Content-Type": file.type,
          },
        })
        .then((r) => updateResults(r.data.image));
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
        {inputImageURL && (
          <>
            <div className={styles.shadow} />
            <Image
              src={inputImageURL}
              alt="Загруженное изображение"
              fill={true}
              style={{
                objectFit: "cover",
                overflow: "hidden",
                borderRadius: "10%",
              }}
            />
          </>
        )}
      </div>
    </div>
  );
}
