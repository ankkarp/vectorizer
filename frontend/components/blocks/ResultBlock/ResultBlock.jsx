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
    console.log(resDir);
    if (resDir) {
      Object.values(resultObjs).forEach((obj) => {
        try {
          http.get(obj.endpoint).then((r) => {
            console.log(typeof r.data);
            obj.setter(
              URL.createObjectURL(new Blob([r.data], { type: obj.blobType }))
            );
          });
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
          <div className={styles.result}>
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
                  // width: "30vw",
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
          </div>
          {Object.entries(resultObjs).forEach(
            (name, obj) =>
              obj.valueURL && (
                <Image
                  src={obj.valueURL}
                  alt={name}
                  fill={true}
                  style={{
                    objectFit: "cover",
                    overflow: "hidden",
                    borderRadius: "10%",
                  }}
                />
              )
          )}

          {/* <div className={styles.result}>
            <div dangerouslySetInnerHTML={{ __html: svgCode }}></div>
          </div> */}
        </>
      )}
    </div>
  );
};

export default ResultBlock;
