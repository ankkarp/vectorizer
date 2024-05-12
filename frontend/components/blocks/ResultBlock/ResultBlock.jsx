import styles from "./ResultBlock.module.css";
import TextField from "@mui/material/TextField";

const ResultBlock = () => {
  return (
    <div className={styles.container}>
      <TextField id="outlined-basic" label="Outlined" variant="outlined" />
    </div>
  );
};

export default ResultBlock;
