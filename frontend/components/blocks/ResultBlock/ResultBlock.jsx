import { useState } from "react";
import styles from "./ResultBlock.module.css";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Save from "@mui/icons-material/SaveAlt";
import ContentCopy from "@mui/icons-material/ContentCopy";
import { Tooltip } from "@mui/material";
import Image from "next/image";

const ResultBlock = ({ svgCode, progressGif }) => {
  const [tooltipOpen, setTooltipOpen] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(svgCode).then(() => {
      setTooltipOpen(true);
      setTimeout(() => setTooltipOpen(false), 2000); // Hide tooltip after 2 seconds
    });
  };

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
                "& .MuiInputBase-input": {
                  color: "var(--text-clr)", // Ensure the text color is white
                  "-webkit-text-fill-color": "var(--text-clr)",
                  paddingRight: "20px",
                },
                "& .MuiInputBase-root.Mui-disabled": {
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
                  color: "#eeeded",
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
                  color: "var(--result-clr)",
                },
              }}
            />
          </div>
          {/* <div className={styles.result}>
            <div dangerouslySetInnerHTML={{ __html: svgCode }}></div>
          </div> */}
        </>
      )}
    </div>
  );
};

export default ResultBlock;
