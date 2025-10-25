export type PipelineResult = {
  outputUrl: string;
  text: string;
  inputText: string;
};

export function useSpeechPipeline(baseUrl = "http://127.0.0.1:8000") {
  const run = async (audioBlob: Blob): Promise<PipelineResult> => {
    const fd = new FormData();
    fd.append("file", audioBlob, "input.webm"); // server converts to wav
    const res = await fetch(`${baseUrl}/s2s`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      throw new Error(`Pipeline request failed: ${res.status}`);
    }
    const json = await res.json();
    const outputUrl: string | undefined = json?.output_url;
    const text: string = json?.text ?? "";
    const inputText: string = json?.input_text ?? "";
    if (!outputUrl) throw new Error("Missing output_url in response.");
    return { outputUrl: `${baseUrl}${outputUrl}`, text, inputText };
  };

  return { run };
}
