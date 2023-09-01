use std::collections::LinkedList;

/// concrete syntax BNF
/// expr : Int 
///     | expr "+" expr
///     | expr "*" expr
///     | "(" expr ")"

// abstract syntax
// Expr编译期需要知道具体大小，递归定义只能用指针存储
enum Expr {
    Cst(i32),
    Add( Box<Expr>, Box<Expr>),
    Mul(Box<Expr>,Box<Expr>),
}
// interpreter
fn eval(expr: Box<Expr>) -> i32 {
    match *expr {
        Expr::Cst(val) => val,
        Expr::Add(expr1, expr2) => eval(expr1) + eval(expr2),
        Expr::Mul(expr1, expr2) => eval(expr1) * eval(expr2),
    }
}

// stack machine
type Operand = i32;
//use i32 as Operand;
enum Instr {
    Cst(Operand),
    Add,
    Mul,
}

type Instrs = LinkedList<Instr>;
type Stack = LinkedList<i32>;

fn eval_stack(mut instrs: Instrs, mut stk: Stack) -> Operand {
    // linkedList的实现不好进行模式匹配，添加一个辅助函数来简化操作
    let do_op = |mut stk: Stack, op: fn(Operand, Operand) -> Operand| {
        let a = stk.pop_front();
        let b = stk.pop_front();
        match (a, b) {
            (Some(a1), Some(b1)) => {
                stk.push_front(op(a1,b1));
                stk
            },
            _ => panic!()
        }
    };
    match instrs.pop_front() {
        Some(Instr::Cst(val)) => eval_stack(instrs, { stk.push_front(val); stk }),
        Some(Instr::Add) => eval_stack(instrs, {
            do_op(stk, |a,b|a+b)
        }),
        Some(Instr::Mul) => eval_stack(instrs, {
            do_op(stk, |a,b|a*b)
        }),
        None => *stk.front().unwrap()
    }
}

// compiler
fn compile(expr: Box<Expr>) -> Instrs {
    match *expr {
        Expr::Cst(i) => {
            Instrs::from([Instr::Cst(i)])
        },
        Expr::Add(expr1, expr2) => {
            let mut stack1 = compile(expr1);
            let mut stack2 = compile(expr2);
            stack1.append(&mut stack2);
            stack1.push_back(Instr::Add);
            stack1
        },
        Expr::Mul(expr1, expr2) => {
            let mut stack1 = compile(expr1);
            let mut stack2 = compile(expr2);
            stack1.append(&mut stack2);
            stack1.push_back(Instr::Mul);
            stack1
        }
    }
}

// 简化new操作的宏
macro_rules! cst {
    ($e: expr) => {
        Box::new(Expr::Cst($e))
    };
}

macro_rules! add {
    ($($e: expr),*) => {
        Box::new(Expr::Add($($e,)*))
    };
}

macro_rules! mul {
    ($($e: expr),*) => {
        Box::new(Expr::Mul($($e,)*))
    };
}

fn main() {
    // test interpreter
    // (1 + 2) * (3 + 4)
    let test_expr = Box::new(Expr::Mul(
            Box::new(Expr::Add(
                Box::new(Expr::Cst(1)),
                Box::new(Expr::Cst(2))
            )),
            Box::new(Expr::Add(
                Box::new(Expr::Cst(3)),
                Box::new(Expr::Cst(4))
            ))
        ));
    println!("interpreter result:{}", eval(test_expr));

    // test stack machine
    let test_expr = add!(
            mul!(
                cst!(1),
                cst!(2)
            ),
            mul!(
                cst!(3),
                cst!(4)
            )
        );
    let compiled = compile(test_expr);
    let stack = Stack::new();
    println!("stack machine result:{}", eval_stack(compiled, stack))
}
