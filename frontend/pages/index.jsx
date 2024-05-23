import { useState } from "react";
import FileInput from "../components/blocks/FileInput/FileInput";
import ResultBlock from "../components/blocks/ResultBlock/ResultBlock";
import MainLayout from "../components/layouts/MainLayout/MainLayout";

export default function Home() {
  const [svgCode, setSvgCode] = useState(null);
  const [resDir, setResDir] = useState(null);

  return (
    <MainLayout>
      <FileInput setSvgCode={setSvgCode} setResDir={setResDir} />
      <ResultBlock svgCode={svgCode} resDir={resDir} />
    </MainLayout>
  );
}
