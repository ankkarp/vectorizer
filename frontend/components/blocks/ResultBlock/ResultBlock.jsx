import { useState } from "react";
import styles from "./ResultBlock.module.css";
import TextField from "@mui/material/TextField";
import Image from "next/image";

const ResultBlock = ({ svgCode, progressGif }) => {
  return (
    <div className={styles.container}>
      {svgCode && (
        <>
          <div className={styles.result}>
            <TextField
              id="outlined-basic"
              label="Outlined"
              variant="outlined"
              value={svgCode}
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
