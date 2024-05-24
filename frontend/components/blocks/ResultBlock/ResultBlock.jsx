import { useState, useEffect } from "react";
import styles from "./ResultBlock.module.css";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Save from "@mui/icons-material/SaveAlt";
import ContentCopy from "@mui/icons-material/ContentCopy";
import { Tooltip } from "@mui/material";
import { withStyles } from "@mui/material/styles";
import Image from "next/image";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import http from "../../../api/http-common";

const muiStyle = {
  input: {
    color: "var(--text-clr)",
  },
};

const ResultBlock = ({ svgCode, resDir }) => {
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const [contourURL, setContourURL] = useState(null);
  const [gifURL, setGifURL] = useState(null);
  const [svgImageURL, setSvgImageURL] = useState(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(svgCode).then(() => {
      setTooltipOpen(true);
      setTimeout(() => setTooltipOpen(false), 2000); // Hide tooltip after 2 seconds
    });
  };

  const resultObjs = {
    "Визуализация процесса": {
      setter: (v) => setGifURL(v),
      valueURL: gifURL,
      endpoint: `process_gif/${resDir}`,
      blobType: "image/gif",
    },
    Контур: {
      setter: setContourURL,
      valueURL: contourURL,
      endpoint: `contour/${resDir}`,
      blobType: "image/png",
    },
    // Результат: {
    //   setter: setSvgImageURL,
    //   valueURL: svgImageURL,
    //   endpoint: "svg_image/{resdir}",
    //   blobType: "image/png",
    // },
  };

  useEffect(() => {
    console.log(Object.entries(resultObjs));
    if (resDir) {
      Object.values(resultObjs).forEach((obj) => {
        try {
          http.get(obj.endpoint);
        } catch (e) {
          console.log(e);
        }
      });
    }
  }, [resDir]);

  const handleSaveAsSVG = () => {
    const blob = new Blob([svgCode], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "text.svg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className={styles.container}>
      {svgCode && (
        <>
          <TextField
            id="outlined-basic"
            label="Изображение в формате SVG"
            minRows={3}
            variant="outlined"
            value={svgCode}
            multiline
            disabled
            fullWidth
            sx={{
              // input: {
              //   color: "var(--text-clr)",
              // },
              "& .MuiFormControl-fullwidth": {
                "& .MuiTextField-root": {
                  alignItems: "center",
                  width: "fit-content",
                },
              },
              "& .MuiInputBase-input.Mui-disabled": {
                WebkitTextFillColor: "var(--text-clr)", // Example color - choose your own
                "-webkit-opacity": 1, // Ensure consistent opacity across browsers
              },
              "& .MuiInputBase-input": {
                //   color: "var(--text-clr)", // Ensure the text color is white
                //   "-webkit-text-fill-color": "var(--text-clr)",
                paddingRight: "20px",
              },
              "& .MuiInputBase-root.Mui-disabled": {
                // color: "var(--text-clr)",
                "& fieldset": {
                  borderColor: "var(--text-clr)",
                },
              },
              "& .MuiOutlinedInput-root": {
                // "& fieldset": {
                //   borderColor: "var(--accent-clr)",
                // },
                "&:hover fieldset": {
                  borderColor: "var(--accent-clr)", // Optional: Set border color to white on hover
                },
                "&.Mui-focused fieldset": {
                  borderColor: "var(--accent-clr)", // Optional: Set border color to white when focused
                },
              },
            }}
            InputProps={{
              style: {
                color: "var(--text-clr)",
                "-webkit-text-fill-color": "var(--text-clr)",
                width: "30vw",
              },
              endAdornment: (
                <InputAdornment position="end">
                  <Tooltip
                    title="Copied!"
                    open={tooltipOpen}
                    disableHoverListener
                    disableFocusListener
                    disableTouchListener
                    placement="top-start"
                  >
                    <IconButton
                      onClick={handleCopy}
                      sx={{
                        position: "absolute",
                        top: 0,
                        right: 0,
                        margin: "10px",
                      }}
                    >
                      <ContentCopy style={{ color: "var(--copy-clr)" }} />
                    </IconButton>
                  </Tooltip>
                  <IconButton
                    onClick={handleSaveAsSVG}
                    sx={{
                      position: "absolute",
                      bottom: 0,
                      right: 0,
                      margin: "10px",
                    }}
                  >
                    <Save style={{ color: "var(--save-clr)" }} />
                  </IconButton>
                </InputAdornment>
              ),
            }}
            InputLabelProps={{
              style: {
                color: "var(--text-clr)",
              },
            }}
          />
          <div className={styles.grid}>
            <div>
              <div className={styles.fitted}>
                <Image
                  src={`${process.env.NEXT_PUBLIC_SERVER_ADDRESS}/process_gif/${resDir}`}
                  alt="Процесс векторизации"
                  fill={true}
                  style={{
                    objectFit: "cover",
                    overflow: "hidden",
                    borderRadius: "10%",
                  }}
                />
              </div>
              <p>Процесс векторизации</p>
            </div>
            <div>
              <div className={styles.fitted}>
                <Image
                  src={`${process.env.NEXT_PUBLIC_SERVER_ADDRESS}/contour/${resDir}`}
                  alt="Контур"
                  fill={true}
                  style={{
                    objectFit: "cover",
                    overflow: "hidden",
                    borderRadius: "10%",
                  }}
                />
              </div>
              <p>Контур</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ResultBlock;
