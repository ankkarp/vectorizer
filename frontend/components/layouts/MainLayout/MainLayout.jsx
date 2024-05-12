import Header from "../../../components/blocks/Header/Header";
import Head from "next/head";
import { useEffect, useState } from "react";
import styles from "./MainLayout.module.css";

const MainLayout = ({ children, show = true }) => {
  const [active, setActive] = useState(false);

  useEffect(() => {
    setActive(true);
  }, []);

  return (
    <>
      <Head>
        <title>Конвертер SVG</title>
      </Head>
      <div className={`${styles.container} ${active ? styles.active : ""}`}>
        {children}
      </div>
    </>
  );
};

export default MainLayout;
