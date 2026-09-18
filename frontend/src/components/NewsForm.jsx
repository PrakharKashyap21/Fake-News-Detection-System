import React, { useState } from "react";

const NewsForm = ({ onSubmit, isLoading, validationError, setValidationError }) => {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setValidationError("");

    const trimmedTitle = title.trim();
    const trimmedText = text.trim();

    if (!trimmedTitle && !trimmedText) {
      setValidationError("Please enter a headline or article text.");
      return;
    }

    onSubmit({ title: trimmedTitle, text: trimmedText });
  };

  return (
    <form className="news-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="headline-input" className="form-label">
          Headline <span className="label-optional">(Optional)</span>
        </label>
        <input
          id="headline-input"
          type="text"
          className="form-input"
          placeholder="Enter news article headline or title..."
          value={title}
          onChange={(e) => {
            setTitle(e.target.value);
            if (validationError) setValidationError("");
          }}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="article-text-input" className="form-label">
          Article Text <span className="label-required">*</span>
        </label>
        <textarea
          id="article-text-input"
          className="form-textarea"
          rows={7}
          placeholder="Paste full news article text here..."
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            if (validationError) setValidationError("");
          }}
          disabled={isLoading}
        />
      </div>

      {validationError && (
        <div className="validation-alert" role="alert">
          {validationError}
        </div>
      )}

      <button
        type="submit"
        className={`submit-button ${isLoading ? "loading" : ""}`}
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <span className="spinner" aria-hidden="true"></span>
            <span>Analyzing...</span>
          </>
        ) : (
          "Detect News"
        )}
      </button>
    </form>
  );
};

export default NewsForm;
