CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ??
=p??
      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MŠ3   max       Pu?       ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??9X   max       =???      ?  ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>???
=q   max       @FǮz?H     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???G?|    max       @vp??
=p     	`  )?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @*         max       @Q?           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @??           ?  3?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?#?
   max       >???      ?  4?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B5??      ?  5?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?}?   max       B5??      ?  6?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?;[#   max       C?l      ?  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?.?   max       C?g?      ?  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      ?  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      ?  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MŠ3   max       P#??      ?  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??A [?7   max       ?ⲕ?ᰊ      ?  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?u   max       =???      ?  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>???
=q   max       @FǮz?H     	`  >?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???
=p?    max       @vp??
=p     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @&         max       @Q?           x  Q?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @??@          ?  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >      ?  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?!-w1??   max       ????2?W?     ?  T         
            >         /               -               
   3   	                   C            E                           	      	         .   %   F   
      !                        ?   6OB?[N,<?N??LOMk?N6O???Pu? Nqb?N?v}PP?O??8N??PN?`nO??fO??O??HN??NX"jN?lN??HP[?'N???N?4?N?c?N???O?U8N?&zP?O?eN{?N?txO??tO(O4?GO?q?N?-?OU?HNh;ZO?M?\9N???Ok?6N???N?Ntc?PNv_O?B?P4?=N???N???O?6?NJkAM?MŠ3O??O?$OO?N?|O?Z;Os?s??9X?T???D????`B???
??o%   ;o;D??;?o;?o;?`B<o<t?<t?<#?
<T??<T??<T??<?o<?o<?t?<?t?<?t?<??
<??
<??
<?9X<?j<ě?<?/<?`B<??h<?<?<???<???=C?=C?=\)=?P=??='??='??='??=0 ?=49X=49X=49X=8Q?=8Q?=@?=@?=?%=??=?+=?O?=?O?=??-=????????
#04:940#
???V[]abnppnlgbVVVVVVVVRWZ[agtz{ytqg[RRRRRRCCDGMN[gtx~??~tpg[NC??????????????????????????????????????????????/>FMNH/#	??? 
#)&#
???????

???????	
)B[g????tgI50	48BNWgtz???tg[NB>64YU[hptz?????tlh`][YY??????????????????????????
"'#
????ONTnw???????????zaVO???????
'+%$ 
???yz|??????????zyyyyyyldnz??????znllllllll??
"
????????????"$/;HJHD<;/."?????????=@@9???SRPTajmz????ztma^TSS?????????????????????? ???????????????????????????????
#/:92!?????????????????????????????????		?????????)6BHJG?6)???????????????????????  #(/0<BHPLHC</#    ????????????????????	#/255/-)#
	)*1NWWNMB53)(`go?????????????tjg`
	)-5655+)&

	%)57;?<:5)	?|?????????????????"#/<<HU[VULH@</.$),6:6)##05<<B<40#nz???????????{upmon???????????????????????


???????????B=?IUabcb]UIBBBBBBBB????+0:;)??????????	,=BOO[\WOG>7+)	?,4516B[htz?????xhO6,ebddhht~????~wtheeeeSPTU[ansxz}?ztnmaUSS??????????????????????????????????????????????????????)+)!vz????????????????|v???????????????")5BO\bgg[NB5)qmt????????????|ttqq???????

?????????????????????Ļ????????????????????????x?t?m?p?q?x?{???нݽ?????? ???????ݽн˽ннннннн??U?a?n?o?zÃÂ?z?n?a?V?U?P?R?U?U?U?U?U?U???????????????????????????????????????????#?)?6?)????	????????????????????? ??????????ƼƳƱưƴ?????ٿT?`?y?????????????y?`?;??޾????	?)?F?T??'?4?<?5???4?'?$????????????????????????????????????????????????????????$?0?;?C?-?/?'???????????ƟƧƳ????ƁƎƒƌƁ?{?{?u?h?\?C?6?+?+?>?U?f?h?uƁ?s???????z?s?f?b?Z?M?M?H?M?P?Z?f?i?s?s??????!?&? ????????????????????????H?T?]?d?m?q?{?v?m?a?Z?H?;?"???)?/?9?H????5?A?a?g?q?h?Z?N?A??????????????????)?5?[?^?Z?M?)????????????????????a?c?m?z???????z?m?a?]?V?W?`?a?a?a?a?a?a?zÇÓÐÌÏÇÅ?z?z?v?y?z?z?z?z?z?z?z?z?"?$?%?%?"??????"?"?"?"?"?"?"?"?"?"?H?T?a?l?m?u?y?q?m?j?a?T?R?H?H?=?H?H?H?H?H?m?z???????????????????z?T?H???,?;?H???????????????????????????????????????ؿ;?G?O?T?H?G?;?.?"???"?.?/?;?;?;?;?;?;?????????????ļ??????????????????????????*?2?3?*???????????????(?2?4?A?Z?c?f?e?]?A?5????????????"?%?.?7?;?@?;?:?.?"???	???????	??Ěĳ????????????????ĿĳĚč?|?n?o?sČĚ??????????????????׾ʾ?????????????????????(?4?7?4?(?$?(?(?(??????
?????????????????????????????????????޹????3?8?C?E?@?3?'????ܹϹȹŹƹ͹?????????????????s?h?f?Z?U?M?K?M?Z?f?s????4?A?M?Z?Z?Z?Z?M?8?4?(????
????(?4?4???M?f?s?{??x?f?Z?A?4?(?!???"?(?.?4?`?m?y???y?w?p?m?a?`?Z?T?G?F?G?K?T?T?`?`??#?0?<?A?N?U?[?X?U?<?0?#??
???
??ÇÓàåáàÖÓÑÇÁ?~ÇÇÇÇÇÇÇÇ??(?4?5?9?5?4?+?*?(????????????????????????????????????????????????????ּ????????????????????ּӼѼּּּּּּ?????ܻû????????????????ûܻ??????
???????
?????????????
?
?
?
?
?
?Y?e?i?r?r?r?e?Y?P?O?Y?Y?Y?Y?Y?Y?Y?Y?Y?Y?????????????????????????????????????????H?S?O?U?K?/?????????????????????(?;?H?~???????ɺֺ??ֺɺ??????????r?e?Y?S?e?~?л???'?I?j???{?h?Y?M?@?4?????߻ԻĻп????Ŀѿݿݿ??ݿֿѿĿÿ????????????????????????????????????????????????????????G?S?`?y???????????????????????y?`?S?@?GE?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?Eپ????????????׾׾־Ͼ׾??????????????????-?:?=?;?C?:?8?-?(?-?-?-?-?-?-?-?-?-?-?-?????ʼμ??????????ּʼ??????????????????g?s?????????????????s?q?g?_?\?`?g?g?g?g?????????????????????????????????????????s???????????u?s?g?Z?X?N?M?N?S?Z?g?g?s?sD?D?D?D?D?D?D?D?D?D?D?D?D?D?DsDkDnDrD{D?EiEuE?E?E?E?E?E?E?E?E?E?E?E?EuEoEiE\E_Ei * m H  X 1 9 L * 7 b S . < R U 5 n l O H F T X , J ? ' ] j 9 ; E S 6 I 4 9 9 q   ? a 4 6 A I ? H 4 \ A ? f 2 ( P 8  3  ?  u  ?  ?  a      ?  ?  ?  q  ?  ?  K  k  :    ?  h    ?    ?  ?  ?  v    ?  ?  ?  ?  y    ?  ?  ?  ?  k  =  %  ?  ?  ?  =  ?  ?    A  ?    ?  L  5  +  K  M  ?    ?  ???`B?#?
?o<D????o<D??=y?#;??
<49X=L??<??
<T??<?t?<???=]/=??<?/<??
<?o<???=?7L<?/<???<???<???=P?`<???=?E?=]/<???=,1=Ƨ?=49X=?P=P?`=??=8Q?=?w=<j=?P=8Q?=m?h=H?9=D??=D??=?9X=??T=?l?=]/=T??=??w=??=H?9=??=?1=???=\=??T>???>2-B$??B'??B	]B?jByWB??B^
B$??B
;BR'B??B??B!?VBb?B??B?2B ?,BX?B$X?A???BzXA?H?B?cB#!?BX[B??B ?B??B?VB!??B??Bg?B{?BFB#?BS?B?<B ??B?HB9?B%??B?*B?`B$B'?B}?BUB??B?B?B,??B'B5??B?EB?cB^?B?2B
vB?rB$B$?B'?ZB	K?B?,B??B??B??B$?WB??B>?B	JLB??B!޽BUqBJ?B??B ?7B?yB$m?A?}?B?A??B??B#;mB?Bt?B ?JB?>B@B!?FB?B??BBT?B??BH?Bm?B |B}AB?B%PB>?B??B$=?B&?SB??B;?BR?BH?B?eB,??B??B5??B?4B<?BKHB??B
??B??BL@???A,(
A?>A???A?x?B?`Ae??@ˬMA?g?BL?B??AA:!@?I;A???A?+?A?R?A?Y(Aɋ?A?+?A?0?A?_A???Ab??@?R[A??A??iA^ZwA??AO??A4ʡA??m?;[#AC??A9??A<6?Aiv?A?b?Aʅ?A5ch@??A?H@??A?}\???)@???A??U@?@ɡ	Ay(?A??cA??C?lAU?@z??@???A??A??BA???C??pC??4@??A,??A?m?A??A???B"?Aj ?@˩OA??B??B?A@?l@??,A??A?y*A??hA?z?A??A??A???A??nA?u?AaN?@?rmA?}kA???A_?A?AP??A3??A???.?AF4A8??A<??Ai?A?PAʀ?A5$?@???A?@?eA??W???f@??A?|@s<@??`AxA??3A?yC?g?AST?@t*s@??&A??3A???A??C??C? X                     >      	   /               .                  4   
                   D             F                           	      	         .   &   F         "                        ?   6                     7         1               '   %               3               %      %   %         '         !                     "            5   '   /         #            !                                    -                        %   #               +               !         %                                                   )   !                        !               Nhe?N,<?N?˄Nɂ=N6Ot??P#??Nqb?N2??OM?VO??N??PN?`nO??fO???O?%PN??8NX"jN?lN??HP?>N??_NN.~N?c?N???O???N??UOU??O?eN{?Nf?O? O(O4?GO?1_N?-?O>??Nh;ZN?L?M?\9N]axO[?N???N?N?ӳPN"O?iO^cN???N???O?)NJkAM?MŠ3O??O?$OB?;N?|O??Os?s  ?  +  ?  ?  ?  %  ?  V  D  ?  ?    X  ?  n  q  ?  ?  $  P  ?  \  ?  ?  [  ?  ?    ?  Q  ?  ?  ?  ^  ?  ?  ]  ?  ?  <  ]    5  6  ?  ?  ?  ?      ?  =  c      ,  ?  :  ?  мu?T???49X%   ???
%   <T??;o;ě?<?`B;??
;?`B<o<t?<T??<u<e`B<T??<T??<?o<?<??
<???<?t?<??
<ě?<?1=aG?<?j<ě?<???=P?`<??h<?<???<???=o=C?=\)=\)=??=?w='??='??=,1=L??=@?=?9X=49X=8Q?=aG?=@?=@?=?%=??=?+=?\)=?O?=?l?=???

#*+#








V[]abnppnlgbVVVVVVVVY[[bgty{ytmg[[YYYYYYMMLNS[gkstwtrg[NMMMM????????????????????????????????????????????
#/>EID</#???? 
#)&#
?????????????????# "')5BINQXUNJB5)#87BNU[gtx????}tgb[N8YU[hptz?????tlh`][YY??????????????????????????
"'#
????SRXry???????????zaYS??????
$&'#
???{~??????????{{{{{{{{ldnz??????znllllllll??
"
????????????"$/;HJHD<;/."???????+574-???XWamz}?zomiaXXXXXXXX???????????????????????? ???????????????????????????????
#)/762-???????????????????????????????????????????????)6BHJG?6)???????????????????????/&,/8<>HMHH<<///////????????????????????	#/255/-)#
	)*1NWWNMB53)(mgjp??????????????tm
	)-5655+)&


	')/58:=;95)
?|????????????????? #&/8<HUUKH?<//&#  ),6:6)#.009;10#mpz?????????????|vm???????????????????????


???????????D?AIUYa[UIDDDDDDDDDD????????'+23)??? 06BVYTKE=6) MKIMO[[htt?zuthg[OMMebddhht~????~wtheeeeSPTU[ansxz}?ztnmaUSS????????????????????????????????????????????????????????????)+)!vz????????????????|v???????????????#)5BNN[bf[NB5)qmt????????????|ttqq????????

?????????????????????Ļx???????????????{?x?x?x?x?x?x?x?x?x?x?x?нݽ?????? ???????ݽн˽ннннннн??a?j?n?zÂÀ?z?n?a?X?U?R?U?U?a?a?a?a?a?a???????????????????????????????????????????#?)?6?)????	???????????????????????????????????????ƹƷƼ???ٿ`?m?????????????y?`?;?"?? ????.?;?T?`??'?4?<?5???4?'?$????????????????????????????????????????????????????????????$?&?+? ????????????????????uƁƍƉƁ?~?z?y?u?h?\?O?C?6?-?-?6?@?W?u?s???????z?s?f?b?Z?M?M?H?M?P?Z?f?i?s?s??????!?&? ????????????????????????H?T?]?d?m?q?{?v?m?a?Z?H?;?"???)?/?9?H????5?A?Z?e?b?Z?N?A?5?(???????????????)?5?N?[?U?G?5?)?????????????????a?m?z?}?????z?m?a?^?W?X?a?a?a?a?a?a?a?a?zÇÓÐÌÏÇÅ?z?z?v?y?z?z?z?z?z?z?z?z?"?$?%?%?"??????"?"?"?"?"?"?"?"?"?"?H?T?a?l?m?u?y?q?m?j?a?T?R?H?H?=?H?H?H?H?H?T?m?????????????????z?a?T?H?,?+?2?;?H???????????????????????????????????????ؿ;?G?I?Q?G?<?;?/?.?"?.?5?;?;?;?;?;?;?;?;?????????????ļ??????????????????????????*?2?3?*????????????????A?J?S?]?`?_?Z?S?A?5?(??????????"?"?-?.?;?;?;?4?.?"???	??? ?	?? ?"?"ĦĳĿ????????????ĿĳĦĚēčĊđĚĝĦ??????????????????׾ʾ?????????????????????(?4?7?4?(?$?(?(?(??????
??????????????????????????????????????޹????????)?(????????ܹҹϹѹٹܹ辌????????????s?h?f?Z?U?M?K?M?Z?f?s????4?A?M?Z?Z?Z?Z?M?8?4?(????
????(?4?(?4?A?M?f?s?z?~?w?f?Z?M?A?4?#????$?(?`?m?y???y?w?p?m?a?`?Z?T?G?F?G?K?T?T?`?`??#?0?<?@?L?U?X?U?P?I?<?0?#???
?
??ÇÓàåáàÖÓÑÇÁ?~ÇÇÇÇÇÇÇÇ??(?3?4?7?4?3?*?(????? ???????????????????????????????????????????????ּ????????????????ּռҼֻּּּּּּּ???????ܻлû??????????????????ûܻ??
???????
?????????????
?
?
?
?
?
?Y?e?i?r?r?r?e?Y?P?O?Y?Y?Y?Y?Y?Y?Y?Y?Y?Y????????????????????????????????????????????????"?;?H?N?E?:?/????????????????˺~?????????ɺϺպ??????????r?e?Y?]?e?r?~???'?4?6?@?A?D?@?@?4?'??????
???????Ŀѿݿݿ??ݿֿѿĿÿ????????????????????????????????????????????????????????l?y???????????????????y?l?i?`?]?W?`?g?lE?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?Eپ????????????׾׾־Ͼ׾??????????????????-?:?=?;?C?:?8?-?(?-?-?-?-?-?-?-?-?-?-?-?????ʼμ??????????ּʼ??????????????????g?s?????????????????s?q?g?_?\?`?g?g?g?g?????????????????????????????????????????s???????????u?s?g?Z?X?N?M?N?S?Z?g?g?s?sD?D?D?D?D?D?D?D?D?D?D?D?D?D?D~DuDwD~D?D?EiEuE?E?E?E?E?E?E?E?E?E?E?E?EuEoEiE\E_Ei G m 5  X $ / L 1 / _ S . < T U ? n l O F 9 @ X , H C  ] j 5 % E S 3 I - 9 4 q   ? a 4 2 9 B 3 H 4 / A ? f 2 ( N 8  3  ?  u  ?  ?  a  ?  ?  ?  I  ?  G  ?  ?  K    i  ?  ?  h      ?  J  ?  ?  ?  ?  ?  ?  ?  v  ?    ?  ?  ?  ?  k    %  k  \  ?  =  I  ?  ?  $  ?    Y  L  5  +  K  M  ?    1  ?  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  d  K  <  5  .  +  #          ?  ?  ?  ?  ?  ?  ?  u  r  p  m  k  h  f  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  ^  C  "  ?  ?  ?  }  I    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  [  ;    ?  >  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  ^  K  8  $  ?  
      $  #        ?  ?  ?  ?  o  P  2    ?  ?  ?  ,    ?  ?  ?  ?  ?  ?  P    ?  ?  ?  ?  ?  t  )  ?  ?    V  S  P  M  J  G  D  >  7  0  (  !        ?  ?  ?  ?  ?  :  ;  =  >  >  ?  A  C  B  ?  ;  4  .  *  &  -  8  A  G  N    O  h  v  ~  ?    h  `  m  ?  ?  ?  ?  V    ?  \  ?  ?  ?  ?  ?  v  g  [  W  =       ?  ?  ?  ?  k  >  "    ?  \    	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  g  [  N  >     ?   ?  X  P  G  B  >  9  4  /  -  -  -  +  '  !    ?  ?  ?  q    ?  ?  ?  ?  ?  z  f  V  J  =  (    ?  ?  ?  ?  W    ?  j  I  h  m  d  T  >  "    ?  ?  ?  ?  {    ?  C    ?  3  ?    -  Y  n  n  d  U  A  K  @  )    ?  ?  {  
  h  ?  `   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  <    ?  ?  A  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  j  \  R  G  B  E  G  W  m  ?  $  &  '  )  *  +  -  &        ?  ?  ?  ?  ?  ?  ?  ?  z  P  N  M  G  ?  3  %      ?  ?  ?  ?  ?  H    ?  ~  4   ?  ?  c  ?  ?  ?  ?  ?  ?  ?  ?  q  H  $  ?  ?  5  ?  Y  ?  T  I  D  >  J  X  O  >  ,      ?  ?  ?  ?  ?  ?  ?  ?  ~  \  ?  ?  ?  ?  ?  ?  ?  k  Q  5    ?  ?  ?  ?  o  J  <  5  /  ?  ?  ?  ?  ?  ?  ?  s  ]  \  _  ^  G  0  $  )  .  &    	  [  Q  G  =  4  -  '             
    ?  ?    R  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  d  D     ?  ?  ?  ?  q  &  ?  |    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  e  T  @     ?   ?    ?  ?  >  z  ?  ?  ?  
      ?  ?  ?  /  ?    J  G  #  ?  g  F  "  
  =  ?  9  )    ?  ?  ?  ?  l  ,  ?  "  K   L  Q  Q  R  R  R  R  R  S  S  S  R  O  L  I  G  D  A  >  ;  8  t  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  B    ?  ?  r  :    ?  @  ?  9  q  ?  ?  ?  ?  ?  ?  i  7    ?  d  ?  ?  0  C  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  n  J  '    ?  ?  ?  {  T  +  ^  Q  C  4  "      ?  ?  ?  ?  ?  ?  ?  ?  ?  |  z  {  |  ?  ?  ?  ?  ?  ?  |  g  P  5    ?  ?  ?  ?  P    ?  n  /  ?  ?  ?  ?  ?  ?  ?  v  l  [  I  7  "    ?  ?  ?  j     ?  [  ]  \  X  P  G  >  6  -  $      ?  ?  ?  ?  ?  Y    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  Z  B  )    ?  ?  ?  ~  z  ?  ?  ?  v  h  V  D  2      ?  ?  ?  ?  ?  ?  c  %  ?  <  E  N  W  `  i  r  |  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  D  O  Y  ^  a  c  c  c  ^  Y  P  B  4      ?  ?  ?  f  7  ?    ?  ?  ?  r  B    ?  ?  @  ?  ?  8  ,  ?  ?  =  ?  }  5  +  "         ?  ?  ?  ?  ?  ?  b  >    ?  ?  ?  r  F  6  1  ,  %        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  f  S  ;      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  V  0    ?  ?  V    ?  c  ?  H  ?  ?  ?  ?  ?  p  I     ?  ?  ?  ?  ?  q  %  ?  Y  ?  ?  ?  ?  ?  ?  q  Z  `  r  ?  ?  ?  ?  ?  ?  ?  ?  $  ?  $  *  ?      ?  ?  ?  ?  ?  k  M  2    ?  ?  ?  ?  w  Q  *    ?      ?  ?  ?  ?  ?  o  V  >  &    ?  ?  ?  ?  ?  ?  ?  ?  ?     |  u  g  ^  ?  ?  ?  t  f  X  F  #  ?  ?  ~  ;  ?  t  =  B  F  I  F  D  6    ?  ?  o     ?  }  '  ?  t    ?  T  c  c  c  b  b  b  a  a  a  `  X  I  9  )    
  ?  ?  ?  ?    &  0  9  B  L  U  ^  h  q  y    ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  i  F     ?  ?  ?  ?  N    ?  ?  ?  w  &  ?  ,      ?  ?  ?  ?  ?  r  Q  +    ?  ?  k  3  ?  ?  ?  C  ?  ?  ?  u  c  P  =  &    ?  ?  q  4  ?  ?  Q  ?  ?  *  ?  :  $    ?  ?  ?  ?  ?    ^  ;    ?  ?  ?  ?  _  Q  *  ?  o  ?  ?  ?  ?  ?  ?  ?  u  H    ?  F  ?  ?  X  ?  d  ?  0  ?  ?  ?  v  I    
?  
?  
p  
(  	?  	y  	  ?  ?  4  j  ?  ?  ?