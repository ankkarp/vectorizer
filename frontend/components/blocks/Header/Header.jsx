import UploadIcon from "../../icons/UploadIcon/UploadIcon";
import styles from "./Header.module.css";

const Header = () => {
  return (
    <div className={styles.header}>
      <UploadIcon /> 
    </div>
  );
};

export default Header;