"""
numerical_calculator_full.py
All-in-one Numerical Methods GUI (Tkinter)
Author: Provided to user
Requirements: numpy, sympy, matplotlib
Run: python numerical_calculator_full.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import isfinite

# ---------------------------
# Numerical method functions
# ---------------------------

def parse_func(expr_str):
    x = sp.symbols('x')
    expr_str = expr_str.replace('^', '**')
    expr = sp.sympify(expr_str)
    f = sp.lambdify(x, expr, modules=["numpy", "math"])
    return f, expr

# Root finding
def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    fa = f(a); fb = f(b)
    steps = []
    if np.isnan(fa) or np.isnan(fb) or fa*fb > 0:
        raise ValueError("Invalid interval: f(a) and f(b) must have opposite signs and be finite.")
    for i in range(1, max_iter+1):
        c = (a+b)/2.0
        fc = f(c)
        steps.append((i, a, b, c, fc))
        if not isfinite(fc):
            raise ValueError("Function produced non-finite value during iterations.")
        if abs(fc) <= tol or (b-a)/2 <= tol:
            return c, steps
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, steps

def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    steps = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        steps.append((i, x, fx, dfx))
        if dfx == 0 or not isfinite(dfx):
            raise ValueError("Zero or non-finite derivative encountered.")
        x_new = x - fx/dfx
        if not isfinite(x_new):
            raise ValueError("Iteration diverged to non-finite value.")
        if abs(x_new - x) < tol:
            return x_new, steps
        x = x_new
    return x, steps

# Integration
def trapezoidal_rule(f, a, b, n=100):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b-a)/n
    return (h/2)*(y[0] + 2*np.sum(y[1:-1]) + y[-1])

def simpson_rule(f, a, b, n=100):
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b-a)/n
    return (h/3) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

# Differentiation (finite differences)
def forward_diff(f, x, h=1e-5):
    return (f(x+h) - f(x))/h

def backward_diff(f, x, h=1e-5):
    return (f(x) - f(x-h))/h

def central_diff(f, x, h=1e-5):
    return (f(x+h) - f(x-h))/(2*h)

# Matrix solvers
def gauss_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    for i in range(n):
        # pivot
        pivot = np.argmax(np.abs(A[i:, i])) + i
        if A[pivot, i] == 0:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            b[[i, pivot]] = b[[pivot, i]]
        for j in range(i+1, n):
            factor = A[j,i]/A[i,i]
            A[j,i:] -= factor * A[i,i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:]))/A[i,i]
    return x

def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2)/A[i,i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k+1
    return x, max_iter

# Interpolation
def lagrange_interp(x_pts, y_pts, x):
    x_pts = np.array(x_pts); y_pts = np.array(y_pts)
    n = len(x_pts)
    total = 0.0
    for i in range(n):
        term = y_pts[i]
        for j in range(n):
            if i != j:
                term *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        total += term
    return total

def newton_divided_diff(x_pts, y_pts, x):
    n = len(x_pts)
    dd = [yi for yi in y_pts]
    coeffs = [dd[0]]
    for level in range(1, n):
        for i in range(n-level):
            dd[i] = (dd[i+1] - dd[i])/(x_pts[i+level] - x_pts[i])
        coeffs.append(dd[0])
    # evaluate
    result = coeffs[0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_pts[i-1])
        result += coeffs[i]*prod
    return result

# Linear regression (least squares)
def linear_regression(xs, ys):
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    return m, c

# ---------------------------
# GUI helpers (hover, layout helpers)
# ---------------------------

BTN_BG = "#0288d1"
BTN_HOVER = "#03a9f4"
PANEL_BG = "#ffffff"
APP_BG = "#e0f7fa"

def btn_on_enter(e):
    e.widget.configure(bg=BTN_HOVER)
def btn_on_leave(e):
    e.widget.configure(bg=BTN_BG)

def make_button(parent, text, cmd, width=18):
    b = tk.Button(parent, text=text, bg=BTN_BG, fg="white", bd=0, relief="raised",
                  font=("Segoe UI", 10, "bold"), command=cmd, width=width, padx=6, pady=6)
    b.bind("<Enter>", btn_on_enter); b.bind("<Leave>", btn_on_leave)
    return b

# common plotting helper
def plot_function_and_points(f, x_start, x_end, points=None, title="Function"):
    xs = np.linspace(x_start, x_end, 400)
    ys = f(xs)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, color='black', linewidth=0.8)
    if points:
        px = [p[0] for p in points]; py = [p[1] for p in points]
        plt.scatter(px, py, color='red', zorder=5)
    plt.title(title)
    plt.legend()
    plt.show()

# ---------------------------
# GUI: Main Dashboard and Toplevels for each calculator
# ---------------------------

class NumericalApp:
    def __init__(self, master):
        self.master = master
        master.title("Numerical Methods Hub")
        master.geometry("820x640")
        master.configure(bg=APP_BG)
        self._build_dashboard()

    def _build_dashboard(self):
        header = tk.Label(self.master, text="Numerical Methods Calculator", bg=APP_BG,
                          fg="#0288d1", font=("Segoe UI", 20, "bold"))
        header.pack(pady=(24,12))

        desc = tk.Label(self.master, text="Choose a calculator category", bg=APP_BG, fg="#333",
                        font=("Segoe UI", 12))
        desc.pack(pady=(0,18))

        # grid frame for buttons (2 columns)
        frame = tk.Frame(self.master, bg=APP_BG)
        frame.pack(pady=6)

        buttons = [
            ("Basic Calculator", self.open_basic),
            ("Root Finding", self.open_root_finder),
            ("Integration", self.open_integration),
            ("Differentiation", self.open_differentiation),
            ("Matrix Solvers", self.open_matrix),
            ("Interpolation", self.open_interpolation),
            ("Linear Regression", self.open_regression)
        ]

        # create grid nicely spaced
        rows = (len(buttons)+1)//2
        idx = 0
        for r in range(rows):
            for c in range(2):
                if idx >= len(buttons): break
                text, cmd = buttons[idx]
                b = make_button(frame, text, cmd, width=28)
                b.grid(row=r, column=c, padx=18, pady=12, sticky="nsew")
                idx += 1

        # footer area for credits / small help
        footer = tk.Label(self.master, text="Tip: click a calculator to open it in a new window.", bg=APP_BG, fg="#666",
                          font=("Segoe UI", 10))
        footer.pack(pady=(20,8))

    # ---------------- Basic with on-screen keyboard ----------------
    def open_basic(self):
        win = tk.Toplevel(self.master); win.title("Basic Calculator"); win.geometry("420x540"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Basic Calculator", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 14, "bold")).pack(pady=12)
        entry = tk.Entry(win, font=("Segoe UI", 18), bd=4, relief="solid", justify='right')
        entry.pack(padx=18, pady=(0,10), fill='x')

        # result label
        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333", font=("Segoe UI", 12))
        res_lbl.pack(pady=(0,6))

        # keyboard layout
        keys = [
            ['7','8','9','/','C'],
            ['4','5','6','*','('],
            ['1','2','3','-',')'],
            ['0','.','=','+','%'],
        ]

        def press(k):
            if k == "C":
                entry.delete(0, tk.END); res_lbl.config(text="")
            elif k == "=":
                expr = entry.get().strip()
                if not expr:
                    return
                try:
                    # use sympy for safe parse and evaluate numeric
                    f, _ = parse_func(expr)  # f expects x input; but sympy lambdify returns array func if x vector used
                    # attempt numeric evaluation: if expression contains 'x' treat as invalid for basic
                    if 'x' in expr:
                        res_lbl.config(text="Use numeric expressions (no 'x').")
                        return
                    # fallback safe eval with numpy allowed
                    val = float(sp.N(sp.sympify(expr)))
                    res_lbl.config(text=f"Result: {val}")
                except Exception:
                    try:
                        val = eval(expr, {"__builtins__":None}, {"np":np, "sin":np.sin, "cos":np.cos, "tan":np.tan, "exp":np.exp, "log":np.log})
                        res_lbl.config(text=f"Result: {val}")
                    except Exception as e:
                        res_lbl.config(text="Error")
            else:
                entry.insert(tk.END, k)

        for row in keys:
            fr = tk.Frame(win, bg=PANEL_BG)
            fr.pack(padx=12, pady=6, fill='x')
            for k in row:
                btn = tk.Button(fr, text=k, bg=BTN_BG, fg="white", font=("Segoe UI", 12, "bold"),
                                width=6, height=2, command=lambda kk=k: press(kk))
                btn.pack(side='left', padx=6)
                btn.bind("<Enter>", btn_on_enter); btn.bind("<Leave>", btn_on_leave)

        # extra function buttons
        extra_fr = tk.Frame(win, bg=PANEL_BG)
        extra_fr.pack(padx=12, pady=8, fill='x')
        for k in ["sin(","cos(","tan(","**2","**3"]:
            btn = tk.Button(extra_fr, text=k, bg="#6cace4", fg="white", font=("Segoe UI", 11), width=8,
                            command=lambda kk=k: entry.insert(tk.END, kk))
            btn.pack(side='left', padx=6)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#5aa7e9"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#6cace4"))

        # clear & close
        bottom = tk.Frame(win, bg=PANEL_BG)
        bottom.pack(fill='x', padx=18, pady=14)
        b_clear = tk.Button(bottom, text="Clear", bg="#ff6b6b", fg="white", command=lambda: (entry.delete(0, tk.END), res_lbl.config(text="")))
        b_clear.pack(side='left', padx=6)
        b_close = tk.Button(bottom, text="Close", bg="#777", fg="white", command=win.destroy)
        b_close.pack(side='right', padx=6)
        for b in (b_clear, b_close): b.bind("<Enter>", btn_on_enter); b.bind("<Leave>", btn_on_leave)

    # ---------------- Root Finder (Bisection & Newton) ----------------
    def open_root_finder(self):
        win = tk.Toplevel(self.master); win.title("Root Finding"); win.geometry("760x520"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Root Finding", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=10)

        topfr = tk.Frame(win, bg=PANEL_BG); topfr.pack(padx=12, pady=6, fill='x')
        tk.Label(topfr, text="f(x) =", bg=PANEL_BG, fg="#333", font=("Segoe UI",12)).grid(row=0, column=0, sticky='w')
        func_entry = tk.Entry(topfr, width=40, font=("Segoe UI",12), bd=3, relief='solid'); func_entry.grid(row=0, column=1, padx=8)
        func_entry.insert(0, "x**3 - 2*x - 5")

        # bisection inputs
        tk.Label(topfr, text="a", bg=PANEL_BG).grid(row=1, column=0, sticky='e', pady=(8,0))
        a_entry = tk.Entry(topfr, width=12); a_entry.grid(row=1, column=1, sticky='w')
        a_entry.insert(0, "1")
        tk.Label(topfr, text="b", bg=PANEL_BG).grid(row=1, column=1, sticky='e')
        b_entry = tk.Entry(topfr, width=12); b_entry.grid(row=1, column=2, sticky='w', padx=(6,0))
        b_entry.insert(0, "3")

        # newton input
        tk.Label(topfr, text="x0 (Newton)", bg=PANEL_BG).grid(row=2, column=0, sticky='e', pady=(6,0))
        x0_entry = tk.Entry(topfr, width=12); x0_entry.grid(row=2, column=1, sticky='w')
        x0_entry.insert(0, "2")

        # options
        tol_entry = tk.Entry(topfr, width=12); tol_entry.grid(row=3, column=1, sticky='w', pady=(8,0))
        tol_entry.insert(0, "1e-8")
        tk.Label(topfr, text="tol:", bg=PANEL_BG).grid(row=3, column=0, sticky='e', pady=(8,0))
        maxit_entry = tk.Entry(topfr, width=12); maxit_entry.grid(row=3, column=2, sticky='w', pady=(8,0))
        maxit_entry.insert(0, "50")
        tk.Label(topfr, text="max iter:", bg=PANEL_BG).grid(row=3, column=1, sticky='e')

        # result & steps
        steps_box = tk.Text(win, height=12, font=("Segoe UI", 10)); steps_box.pack(padx=12, pady=10, fill='both')

        # solver functions
        def run_bisection():
            expr = func_entry.get().strip()
            try:
                f, sym_expr = parse_func(expr)
                a = float(a_entry.get()); b = float(b_entry.get())
                tol = float(tol_entry.get()); miter = int(maxit_entry.get())
                root, steps = bisection_method(f, a, b, tol=tol, max_iter=miter)
                steps_box.delete(1.0, tk.END)
                steps_box.insert(tk.END, f"Bisection root ≈ {root}\n\nIterations:\n")
                for it,a_,b_,c_,fc in steps:
                    steps_box.insert(tk.END, f"iter {it}: a={a_:.6g}, b={b_:.6g}, c={c_:.6g}, f(c)={fc:.6g}\n")
                # plot
                plot_function_and_points(f, a-1, b+1, points=[(root, f(root))], title="Bisection Plot")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def run_newton():
            expr = func_entry.get().strip()
            try:
                f, sym_expr = parse_func(expr)
                x0 = float(x0_entry.get())
                tol = float(tol_entry.get()); miter = int(maxit_entry.get())
                x = sp.symbols('x')
                dexpr = sp.diff(sym_expr, x)
                df = sp.lambdify(x, dexpr, modules=["numpy", "math"])
                root, steps = newton_method(f, df, x0, tol=tol, max_iter=miter)
                steps_box.delete(1.0, tk.END)
                steps_box.insert(tk.END, f"Newton root ≈ {root}\n\nIterations:\n")
                for it, xn, fx, dfx in steps:
                    steps_box.insert(tk.END, f"iter {it}: x_n={xn:.9g}, f(x)={fx:.6g}, f'(x)={dfx:.6g}\n")
                plot_function_and_points(f, x0-4, x0+4, points=[(root, f(root))], title="Newton Plot")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        # buttons
        btn_frame = tk.Frame(win, bg=PANEL_BG); btn_frame.pack(pady=6)
        b1 = make_button(btn_frame, "Run Bisection", run_bisection)
        b1.grid(row=0, column=0, padx=8, pady=6)
        b2 = make_button(btn_frame, "Run Newton", run_newton)
        b2.grid(row=0, column=1, padx=8, pady=6)
        b_close = make_button(btn_frame, "Close", win.destroy); b_close.grid(row=0, column=2, padx=8, pady=6)

    # ---------------- Integration (Trapezoid & Simpson) ----------------
    def open_integration(self):
        win = tk.Toplevel(self.master); win.title("Integration"); win.geometry("680x480"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Numerical Integration", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=8)
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=12, pady=6, fill='x')
        tk.Label(fr, text="f(x) =", bg=PANEL_BG).grid(row=0, column=0, sticky='w')
        func_entry = tk.Entry(fr, width=40); func_entry.grid(row=0, column=1, padx=8)
        func_entry.insert(0, "sin(x)")
        tk.Label(fr, text="a,b:", bg=PANEL_BG).grid(row=1, column=0, sticky='e', pady=(8,0))
        ab_entry = tk.Entry(fr, width=20); ab_entry.grid(row=1, column=1, sticky='w', pady=(8,0))
        ab_entry.insert(0, "0,3.1416")
        n_entry = tk.Entry(fr, width=10); n_entry.grid(row=1, column=2, padx=10)
        n_entry.insert(0, "100")

        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333", font=("Segoe UI", 12)); res_lbl.pack(pady=8)
        steps_box = tk.Text(win, height=10); steps_box.pack(padx=12, pady=6, fill='both')

        def run_trap():
            try:
                f, sym = parse_func(func_entry.get())
                a,b = map(float, ab_entry.get().split(","))
                n = int(n_entry.get())
                val = trapezoidal_rule(f, a, b, n=n)
                res_lbl.config(text=f"Trapezoidal ≈ {val}")
                steps_box.delete(1.0, tk.END)
                steps_box.insert(tk.END, f"n={n}, interval=[{a},{b}] result={val}\n")
                x_vals = np.linspace(a,b,400); y_vals = f(x_vals)
                plt.plot(x_vals, y_vals, label="f(x)"); plt.fill_between(x_vals, y_vals, alpha=0.3)
                plt.title("Trapezoidal Rule"); plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def run_simpson():
            try:
                f, sym = parse_func(func_entry.get())
                a,b = map(float, ab_entry.get().split(","))
                n = int(n_entry.get())
                val = simpson_rule(f, a, b, n=n)
                res_lbl.config(text=f"Simpson ≈ {val}")
                steps_box.delete(1.0, tk.END)
                steps_box.insert(tk.END, f"n={n}, interval=[{a},{b}] result={val}\n")
                x_vals = np.linspace(a,b,400); y_vals = f(x_vals)
                plt.plot(x_vals, y_vals, label="f(x)"); plt.fill_between(x_vals, y_vals, alpha=0.3)
                plt.title("Simpson Rule"); plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_fr = tk.Frame(win, bg=PANEL_BG); btn_fr.pack(pady=6)
        make_button(btn_fr, "Trapezoidal", run_trap).grid(row=0, column=0, padx=8)
        make_button(btn_fr, "Simpson", run_simpson).grid(row=0, column=1, padx=8)
        make_button(btn_fr, "Close", win.destroy).grid(row=0, column=2, padx=8)

    # ---------------- Differentiation ----------------
    def open_differentiation(self):
        # older name - keep compatibility
        self.open_differentiation_window()

    def open_differentiation_window(self):
        win = tk.Toplevel(self.master); win.title("Differentiation"); win.geometry("660x420"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Numerical Differentiation", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=8)
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=12, pady=6, fill='x')
        tk.Label(fr, text="f(x) =", bg=PANEL_BG).grid(row=0, column=0, sticky='w')
        func_entry = tk.Entry(fr, width=40); func_entry.grid(row=0, column=1, padx=8)
        func_entry.insert(0, "x**2")
        tk.Label(fr, text="x:", bg=PANEL_BG).grid(row=1, column=0, sticky='e', pady=(8,0))
        x_entry = tk.Entry(fr, width=12); x_entry.grid(row=1, column=1, sticky='w', pady=(8,0))
        x_entry.insert(0, "2")
        tk.Label(fr, text="h:", bg=PANEL_BG).grid(row=1, column=2, sticky='e')
        h_entry = tk.Entry(fr, width=10); h_entry.grid(row=1, column=3, sticky='w'); h_entry.insert(0, "1e-5")

        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333", font=("Segoe UI", 12)); res_lbl.pack(pady=8)

        def run_forward():
            try:
                f, _ = parse_func(func_entry.get())
                x0 = float(x_entry.get()); h = float(h_entry.get())
                val = forward_diff(f, x0, h=h)
                res_lbl.config(text=f"Forward ≈ {val}")
                # plot tangent
                xs = np.linspace(x0-3, x0+3, 300); ys = f(xs)
                tangent = val*(xs - x0) + f(x0)
                plt.plot(xs, ys, label="f(x)"); plt.plot(xs, tangent, "--r", label="Tangent")
                plt.scatter([x0],[f(x0)], color='red'); plt.title("Forward Difference"); plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        def run_backward():
            try:
                f, _ = parse_func(func_entry.get())
                x0 = float(x_entry.get()); h = float(h_entry.get())
                val = backward_diff(f, x0, h=h)
                res_lbl.config(text=f"Backward ≈ {val}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        def run_central():
            try:
                f, _ = parse_func(func_entry.get())
                x0 = float(x_entry.get()); h = float(h_entry.get())
                val = central_diff(f, x0, h=h)
                res_lbl.config(text=f"Central ≈ {val}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_fr = tk.Frame(win, bg=PANEL_BG); btn_fr.pack(pady=6)
        make_button(btn_fr, "Forward", run_forward).grid(row=0, column=0, padx=6)
        make_button(btn_fr, "Backward", run_backward).grid(row=0, column=1, padx=6)
        make_button(btn_fr, "Central", run_central).grid(row=0, column=2, padx=6)
        make_button(btn_fr, "Close", win.destroy).grid(row=0, column=3, padx=6)

    # ---------------- Matrix solvers ----------------
    def open_matrix(self):
        win = tk.Toplevel(self.master); win.title("Matrix Solvers"); win.geometry("740x520"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Matrix Solvers", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=8)
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=12, pady=6, fill='x')
        tk.Label(fr, text="Enter matrix A (rows separated by ;, elements by , )", bg=PANEL_BG).grid(row=0, column=0, sticky='w')
        A_entry = tk.Entry(fr, width=60); A_entry.grid(row=1, column=0, columnspan=3, pady=6)
        A_entry.insert(0, "2,1;5,7")
        tk.Label(fr, text="Enter vector b (comma separated)", bg=PANEL_BG).grid(row=2, column=0, sticky='w')
        b_entry = tk.Entry(fr, width=40); b_entry.grid(row=3, column=0, pady=6)
        b_entry.insert(0, "11,13")
        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333"); res_lbl.pack(pady=8)

        def parse_A(Astr):
            rows = [r.strip() for r in Astr.strip().split(';') if r.strip()]
            mat = [list(map(float, r.split(','))) for r in rows]
            return np.array(mat)

        def run_gauss():
            try:
                A = parse_A(A_entry.get()); b = np.array(list(map(float, b_entry.get().split(','))))
                x = gauss_elimination(A, b)
                res_lbl.config(text=f"Gauss solution: {x}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def run_gauss_seidel():
            try:
                A = parse_A(A_entry.get()); b = np.array(list(map(float, b_entry.get().split(','))))
                x, it = gauss_seidel(A, b)
                res_lbl.config(text=f"Gauss-Seidel solution: {x} (iters: {it})")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def run_numpy_solve():
            try:
                A = parse_A(A_entry.get()); b = np.array(list(map(float, b_entry.get().split(','))))
                x = np.linalg.solve(A, b)
                res_lbl.config(text=f"Numpy solve: {x}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_fr = tk.Frame(win, bg=PANEL_BG); btn_fr.pack(pady=6)
        make_button(btn_fr, "Gauss Elimination", run_gauss).grid(row=0, column=0, padx=6)
        make_button(btn_fr, "Gauss-Seidel", run_gauss_seidel).grid(row=0, column=1, padx=6)
        make_button(btn_fr, "Numpy Solve", run_numpy_solve).grid(row=0, column=2, padx=6)
        make_button(btn_fr, "Close", win.destroy).grid(row=0, column=3, padx=6)

    # ---------------- Interpolation ----------------
    def open_interpolation(self):
        win = tk.Toplevel(self.master); win.title("Interpolation"); win.geometry("760x520"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Interpolation (Lagrange & Newton)", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=8)
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=12, pady=6, fill='x')
        tk.Label(fr, text="x points (comma separated):", bg=PANEL_BG).grid(row=0, column=0, sticky='w')
        xp_entry = tk.Entry(fr, width=50); xp_entry.grid(row=0, column=1, padx=8); xp_entry.insert(0, "0,1,2")
        tk.Label(fr, text="y points (comma separated):", bg=PANEL_BG).grid(row=1, column=0, sticky='w')
        yp_entry = tk.Entry(fr, width=50); yp_entry.grid(row=1, column=1, padx=8); yp_entry.insert(0, "1,3,2")
        tk.Label(fr, text="Query x:", bg=PANEL_BG).grid(row=2, column=0, sticky='e')
        q_entry = tk.Entry(fr, width=20); q_entry.grid(row=2, column=1, sticky='w')

        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333"); res_lbl.pack(pady=8)

        def run_lagrange():
            try:
                xs = list(map(float, xp_entry.get().split(',')))
                ys = list(map(float, yp_entry.get().split(',')))
                q = float(q_entry.get())
                val = lagrange_interp(xs, ys, q)
                res_lbl.config(text=f"Lagrange P({q}) ≈ {val}")
                # plot
                xs_plot = np.linspace(min(xs)-1, max(xs)+1, 300)
                ys_plot = [lagrange_interp(xs, ys, xv) for xv in xs_plot]
                plt.plot(xs_plot, ys_plot, label='Interpolation')
                plt.scatter(xs, ys, color='red', zorder=5)
                plt.scatter([q],[val], color='green', zorder=6)
                plt.title("Lagrange Interpolation")
                plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def run_newton():
            try:
                xs = list(map(float, xp_entry.get().split(',')))
                ys = list(map(float, yp_entry.get().split(',')))
                q = float(q_entry.get())
                val = newton_divided_diff(xs, ys, q)
                res_lbl.config(text=f"Newton P({q}) ≈ {val}")
                xs_plot = np.linspace(min(xs)-1, max(xs)+1, 300)
                ys_plot = [newton_divided_diff(xs, ys, xv) for xv in xs_plot]
                plt.plot(xs_plot, ys_plot, label='Interpolation')
                plt.scatter(xs, ys, color='red', zorder=5)
                plt.scatter([q],[val], color='green', zorder=6)
                plt.title("Newton Interpolation")
                plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_fr = tk.Frame(win, bg=PANEL_BG); btn_fr.pack(pady=6)
        make_button(btn_fr, "Lagrange", run_lagrange).grid(row=0, column=0, padx=6)
        make_button(btn_fr, "Newton", run_newton).grid(row=0, column=1, padx=6)
        make_button(btn_fr, "Close", win.destroy).grid(row=0, column=2, padx=6)

    # ---------------- Linear regression ----------------
    def open_regression(self):
        win = tk.Toplevel(self.master); win.title("Linear Regression"); win.geometry("700x480"); win.configure(bg=PANEL_BG)
        tk.Label(win, text="Linear Regression (Least Squares)", bg=PANEL_BG, fg="#0288d1", font=("Segoe UI", 16, "bold")).pack(pady=8)
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=12, pady=6, fill='x')
        tk.Label(fr, text="x values (comma):", bg=PANEL_BG).grid(row=0, column=0, sticky='w')
        xs_entry = tk.Entry(fr, width=50); xs_entry.grid(row=0, column=1, padx=8); xs_entry.insert(0, "0,1,2,3")
        tk.Label(fr, text="y values (comma):", bg=PANEL_BG).grid(row=1, column=0, sticky='w')
        ys_entry = tk.Entry(fr, width=50); ys_entry.grid(row=1, column=1, padx=8); ys_entry.insert(0, "1,3,2,5")
        res_lbl = tk.Label(win, text="", bg=PANEL_BG, fg="#333"); res_lbl.pack(pady=8)

        def run_reg():
            try:
                xs = np.array(list(map(float, xs_entry.get().split(','))))
                ys = np.array(list(map(float, ys_entry.get().split(','))))
                m, c = linear_regression(xs, ys)
                res_lbl.config(text=f"y = {m:.6g} x + {c:.6g}")
                # plot
                xs_plot = np.linspace(xs.min()-1, xs.max()+1, 200)
                plt.scatter(xs, ys, label='Data')
                plt.plot(xs_plot, m*xs_plot + c, color='red', label='Fit')
                plt.title("Linear Regression"); plt.legend(); plt.show()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn_fr = tk.Frame(win, bg=PANEL_BG); btn_fr.pack(pady=6)
        make_button(btn_fr, "Fit", run_reg).grid(row=0, column=0, padx=6)
        make_button(btn_fr, "Close", win.destroy).grid(row=0, column=1, padx=6)

# ---------------------------
# Run application
# ---------------------------

def main():
    root = tk.Tk()
    app = NumericalApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
