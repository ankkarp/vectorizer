import { useState } from "react";
import FileInput from "../components/blocks/FileInput/FileInput";
import ResultBlock from "../components/blocks/ResultBlock/ResultBlock";
import UploadIcon from "../components/icons/UploadIcon/UploadIcon";
import MainLayout from "../components/layouts/MainLayout/MainLayout";

export default function Home() {
  const [svgCode, setSvgCode] = useState(null);
  const [progressGif, setProgressGif] = useState(null);

  return (
    <MainLayout>
      <FileInput setSvgCode={setSvgCode} setProgressGif={setProgressGif} />
      <ResultBlock svgCode={svgCode} progressGif={progressGif} />
    </MainLayout>
  );
}
