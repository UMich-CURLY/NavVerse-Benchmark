export default function Home() {
  return (
    <main className="phone-host">
      <iframe
        className="phone-frame"
        src="/phone.html"
        title="NavVerse Real-World Experiment Server"
        allow="camera; accelerometer; gyroscope; magnetometer; screen-wake-lock"
      />
    </main>
  );
}
