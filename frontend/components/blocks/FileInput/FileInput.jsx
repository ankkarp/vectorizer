import styles from "./FileInput.module.css";

import { useState, useRef } from "react";
import UploadIcon from "../../icons/UploadIcon/UploadIcon";
import http from "../../../api/http-common";
import Router from "next/router";
import LoadingIcon from "../../icons/LoadingIcon/LoadingIcon";
import Image from "next/image";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";

export default function FileInput({ setSvgCode, setResDir }) {
  const [loading, setLoading] = useState(false);
  const [inputImageURL, setInputImageURL] = useState(null);
  const [file, setFile] = useState(null);
  const [epochs, setEpochs] = useState(100);
  const inputRef = useRef();

  const updateResults = (svgCode, resDir) => {
    setSvgCode(svgCode);
    setResDir(resDir);
    setLoading(false);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setInputImageURL(URL.createObjectURL(e.target.files[0]));
  };

  const handleClear = (e) => {
    setFile(null);
    setInputImageURL(null);
    setSvgCode(null);
    setResDir(null);
  };

  const handleSubmit = () => {
    let formData = new FormData();
    console.log(typeof file);
    formData.append("image", file);
    try {
      http
        .post(`upload/?max_epochs=${epochs}`, formData, {
          headers: {
            "Content-Type": file.type,
          },
        })
        .then((r) => updateResults(r.data.image, r.data.resdir));
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
          !inputImageURL && (
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
          )
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
      <div className={styles.epoch}>
        <p>Кол-во эпох</p>
        <Slider
          aria-label="Кол-во эпох"
          defaultValue={100}
          valueLabelDisplay="auto"
          shiftStep={30}
          step={100}
          marks
          min={100}
          max={1000}
          // disabled={loading}
          onChange={(_, newValue) => {
            console.log(newValue);
            console.log(typeof newValue);
            setEpochs(newValue);
          }}
          color="warning"
        />
      </div>
      {inputImageURL && (
        <Stack direction="row" spacing={2}>
          <Button onClick={handleSubmit} style={{ color: "var(--copy-clr)" }}>
            Запустить
          </Button>
          <Button onClick={handleClear} style={{ color: "var(--copy-clr)" }}>
            Очистить
          </Button>
        </Stack>
      )}
    </div>
  );
}
