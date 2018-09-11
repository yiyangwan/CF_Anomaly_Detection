clear
close
x = [0:.05:10];
chi = normpdf(x,0,1);
norm2 = normpdf(x,3,1.5);
x0 = 1.4;

% plotting
hold on
plot(x,norm2, 'LineWidth', 2,'Color',[0.8,0.3,0.0]);
plot(x,chi, 'LineWidth', 2,'Color','b');
area(x(x>=x0),chi(x>=x0),'FaceColor',[0.5 0.5 0.5]);

xticks([1,x0,3,5,7])
set(gca,'TickLabelInterpreter','latex');
xticklabels({'1','$\gamma$','3','5','7'})



xlim([0,7]),
ylim([0,0.5]);

hold off

legend('Unknown Abnormal Distribution','Normal Distribution','Interpreter','latex')