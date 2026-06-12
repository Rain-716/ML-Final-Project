import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from utils import load_dataset

def evaluate(name, model, X_test, y_test, labels, out_dir):
    y_pred = model.predict(X_test)
    metrics = {
        'model': name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
    }
    (out_dir / f'{name}_report.txt').write_text(classification_report(y_test, y_pred, target_names=labels, zero_division=0), encoding='utf-8')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, values_format='d')
    ax.set_title(f'Confusion Matrix - {name}')
    plt.tight_layout(); fig.savefig(out_dir / f'{name}_confusion_matrix.png', dpi=160); plt.close(fig)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_source', default='sample', choices=['sample','empathetic'])
    ap.add_argument('--max_samples', type=int, default=8000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'outputs'; out_dir.mkdir(exist_ok=True)
    df = load_dataset(args.data_source, args.max_samples)
    le = LabelEncoder(); y = le.fit_transform(df['label'])
    n_classes = len(set(y))
    test_size = max(0.3 if len(df) < 100 else 0.2, (n_classes + 1) / len(df))
    X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], y, test_size=test_size, random_state=42, stratify=y)
    labels = list(le.classes_)
    models = {
        'baseline_dummy': Pipeline([('tfidf', TfidfVectorizer()), ('clf', DummyClassifier(strategy='most_frequent'))]),
        'logistic_regression': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        'linear_svm': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)), ('clf', LinearSVC(class_weight='balanced'))]),
        'random_forest': Pipeline([('tfidf', TfidfVectorizer(max_features=3000)), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])
    }
    results=[]
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test, labels, out_dir))
    # GridSearchCV on Logistic Regression
    cv_splits = min(3, np.min(np.bincount(y_train)))
    best_model = models['logistic_regression']
    best_params = {}
    if cv_splits >= 2:
        param_grid = {'tfidf__max_features':[1000,3000,None], 'tfidf__ngram_range':[(1,1),(1,2)], 'clf__C':[0.5,1,2,4]}
        gs = GridSearchCV(models['logistic_regression'], param_grid, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42), scoring='f1_macro', n_jobs=-1)
        gs.fit(X_train, y_train); best_model = gs.best_estimator_; best_params = gs.best_params_
        results.append(evaluate('gridsearch_logistic', best_model, X_test, y_test, labels, out_dir))
    pd.DataFrame(results).to_csv(out_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
    # learning curve for best
    try:
        train_sizes, train_scores, val_scores = learning_curve(best_model, df['text_clean'], y, cv=max(2, cv_splits), scoring='f1_macro', train_sizes=np.linspace(0.4,1.0,4))
        fig, ax = plt.subplots(figsize=(7,5)); ax.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='train'); ax.plot(train_sizes, val_scores.mean(axis=1), marker='o', label='cv')
        ax.set_title('Learning Curve (F1-macro)'); ax.set_xlabel('Training samples'); ax.set_ylabel('F1-macro'); ax.legend(); ax.grid(True, alpha=.3)
        plt.tight_layout(); fig.savefig(out_dir / 'learning_curve.png', dpi=160); plt.close(fig)
    except Exception as e:
        (out_dir/'learning_curve_error.txt').write_text(str(e), encoding='utf-8')
    # feature importance for LR
    try:
        vec = best_model.named_steps['tfidf']; clf = best_model.named_steps['clf']
        feats = np.array(vec.get_feature_names_out())
        rows=[]
        for i, lab in enumerate(labels):
            coef = clf.coef_[i] if clf.coef_.ndim>1 else clf.coef_[0]
            for idx in np.argsort(coef)[-10:][::-1]: rows.append({'label':lab,'feature':feats[idx],'weight':float(coef[idx])})
        pd.DataFrame(rows).to_csv(out_dir/'top_features.csv',index=False,encoding='utf-8-sig')
    except Exception as e: (out_dir/'top_features_error.txt').write_text(str(e), encoding='utf-8')
    joblib.dump({'model':best_model,'label_encoder':le,'responses':df[['text_clean','label','response']].to_dict('records')}, out_dir/'best_model.joblib')
    (out_dir/'best_params.json').write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(results))
    print('Saved model to', out_dir/'best_model.joblib')
if __name__ == '__main__': main()
