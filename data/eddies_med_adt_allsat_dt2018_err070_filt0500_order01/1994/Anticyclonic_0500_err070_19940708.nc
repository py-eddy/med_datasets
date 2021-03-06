CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ??z?G?{       ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??   max       P?F?       ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?q??   max       =?
=       ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?         max       @E?33333     
P   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?\    max       @vp?????     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @"         max       @R@           ?  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?9        max       @??            5?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?T??   max       >M??       6?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??   max       B/=       7?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?
?   max       B/=       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =???   max       C?i8       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =z??   max       C?h!       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??   max       P??       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??b??}W   max       ??0??(?       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?q??   max       =?
=       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?         max       @E?33333     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??\(?    max       @vp?????     
P  L?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @"         max       @R@           ?  V?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?9        max       @??            Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ??   max         ??       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?rn??O?<   max       ??/??v?     ?  Y|                           P         O         :            *      
   !   4                            7            	            ,                  ?            	      7   
      (            
         	   +      1   Nq?YN:0NlliNRqO?uN?(?Ng?N?"?P???NAx?M???P?F?O??OE?P3JP!nOXNg?3Oʪ?N!?SN8X?O???O?;?O?3PPOђN?5M??NK?4O?5NuE?O?G?O???N?M?Ou??OkI
N?9&N"a`O0)BO???N??Nj?AO???OBf<Ol??OͰN??ENۈ?O%?N^?N???Oz+NN??Nk?3O??N??3NҁMNk?SN??N?KOfGN\?sO7??N??.O@??N?zC?q????/???
???㼓t??u?o?o%   ;??
;??
;ě?;ě?;ě?;?`B;?`B;?`B<#?
<#?
<#?
<49X<u<u<u<?o<?o<?C?<?t?<???<???<??
<?1<?1<?j<ě?<???<???<???<???=t?=t?=t?=??=#?
=#?
=,1=,1=,1=0 ?=@?=D??=P?`=T??=T??=]/=]/=q??=q??=?+=?C?=?hs=???=?^5=?^5=Ƨ?=?
=????????????????????-*./<HJHG<0/--------`\ZVVWbgikhhkb``````????????????????????sqt?????????????wvts????????????????????????????????????????????????????????????97GE9=N[gw???????gN9?????????????????????????????/5<N??????????tK/4358BN[gv{tng[MBA=:4??????

 ??????qqz??????????????~zq??????(45BFKB5????????

???????"!")/494/"""""""""???????????????????????????????????????????????????? )5BNV]fg`NB5,)''-<HUar??~naUN<5,'&)/6@BOPTQOICB62-)'XS[cglt???????????hX????????????????????
&%









????????????????????????
?????????6BOUR[^`_[OB)#/8<G</,#???????????????????????????
??????????????????????????????????????RTX^aovz?????}zna^XR????????????????????	

TQ[\ht???????ytmh][Ta^]^anq????????ztnea????????????????????????????????????????????%,/+((("???~{{???????????????????
#*6;<90,#
???????? 
?????????????

??????????????????????????????????????????hkrt???????thhhhhhhh#0<>G><910#zuux}??????????????z# "#/1<AHHHG<5/-###OMKUanoqnfaUOOOOOOOO
	+6BDGJJB<6)1++16<BGGFCB:6111111?????????????????????'%????????????????????????????????????????#;HTabddba^TH;7121/#?????


???????????????$)+.)???516BIOQROB:655555555??????? 

?????????????????????????????????????????????ּӼּټ???????????D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?DӻF?S?_?l?x???????x?l?_?S?F?@?F?F?F?F?F?FčĚģĦĪĦĠĚĕčĉĊčččččččč????????????????ܹ׹׹عܹ?????{ŇŔŠŭŹ??ŹŭšŠŔŇ?{?q?v?{?{?{?{?[?h?t?t?w?v?t?h?c?h?l?h?[?V?[?[?[?[?[?[????????????????????????????????????????B?[?t¸¹¦?g?B?????????????*?6?A?@?6?*?'????????????????????????????????ùøù??????????????????????????????????s?h?Z?N?:?;?Z??????čĚĤĦĨĦĝėčā?n?[?R?N?O?[?h?tāč?????????????????????????????s?q?s???????????????????????????g?N?*?-?5?A?Z?s?????hƎƧƬƥƛƖƗƧƳƹƷƚ?u?_?T?L?O?^?h???????????????????x?l?`?_?^?l?q?x?????????	??"?'?"? ??	???????????????????????/?;?a?m?z???????|?m?a?T?;?2?'?"???"?/?/?<?<?<?9?/?)?#?!??#?#?/?/?/?/?/?/?/?/E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E????0?=?B?I?R?X?U?O?I?=?$????????????????????#?#???????????????????????ѿݿ??????????????ݿѿοĿ????????ĿʿѾM?Z?`?s??????????????f?Z?M?F?B?:?9?A?M???????????????????????????????????????ҿ`?m?p?y?{?y?m?`?Y?\?`?`?`?`?`?`?`?`?`?`?;?G?K?T?U?T?G?;?.?-?.?9?;?;?;?;?;?;?;?;?G?J?T?U?U?T?G?;?.?,?.?;?;?C?G?G?G?G?G?G???	?????	?????????׾ʾ??????þʾ׾????? ?$????
?????????????ùϹܹ????"?'?(?'??????ܹȹ??????þ׾???????????????׾ʾ??????????????ʾ׿???????????????????????????????????????????????????????????????????????N?Z?g?s?u?g?Z?V?N?A?5?(??????(?5?Nìù??????????ùììàÞØßàåìììì?T?`?m?m?n?m?`?T?M?M?T?T?T?T?T?T?T?T?T?T?A?M?Y?Z?e?l?l?f?Z?M?A?4?,?(?$?(?*?4?6?A???ùϹܹ??????ܹӹϹù??????????????????a?n?w?x?s?q?zÂ?z?n?j?d?a?Z?U?Q?U?`?a?aÓÝàêåà×ÓÏÌÇÄÇÈÓÓÓÓÓÓ?#?0?<?I?P?R?O?I?<?0?#?
???????????????#ù??????????????????????????ùìæî÷ù???????ɺֺ??????????ֺɺ???????????????DoD?D?D?D?D?D?D?D?D?D?D?D?D{DpDgDbD_DdDo?ʾ׾??????	???	?????????۾׾ʾ????þʻûлܻ?????	??????????ܻлȻû????ûþ???(?4?A?B?F?D?A?4?(??????????:?F?S?U?V?S?P?F?C?:?8?1?:?:?:?:?:?:?:?:????????
?????????????ּԼҼּ?????????????"?+?+?#???????????????????A?M?Z?f?g?g?f?^?Z?M?M?L?A?8?4?0?4?A?A?A?(?4?A?C?G?B?A?4?/?(?&?$?(?(?(?(?(?(?(?(?r?~???????????ɺԺɺ??????r?e?^?Y?\?e?r?????????????????ٺ????????????????"?/?7?;?>?=?;?8?/?"???????	???"?	??"?$?%?"???	???	?	?	?	?	?	?	?	?	?f?r???????r?p?f?Y?Y?T?Y?b?f?f?f?f?f?f?y???????????z?y?x?v?n?m?y?y?y?y?y?y?y?y?#?)?(?%???
?????????????????????
??#ǈǔǡǭǩǡǕǔǔǈǂǆǈǈǈǈǈǈǈǈ?@?M?T?Y?[?Z?V?M?@?4?)?'????? ?+?4?@????????????????????????EuExE?E?E?E?E?E?E?E?E?E?E?E?EzEqEoEmEjEuE7ECEPEZE\EbE\EPECE7E4E0E7E7E7E7E7E7E7E7 - F ? B % ? k I * 4 ? : : G = N K ? 3 D D  4 0 < ( E b _ # L ( ? B ! E A (   = | D R 3  6 o M > N %  / < D Q O a \ b 0 / Y C > G  q  i  ?  z  V  ?  L  ?  ?  ^  ?  	    ?  ?  ?  <  ~  ?  Q  \  ?  w  #  ?  2  4  !  ?  o  ?  ?  ?    ?  ?  ?  6  s      ?  ?  ?  ?    D    w  ?    ?  ?  ?  z  ?    y  ?  ?  ?  i  ?  J  ?  ??T???o?u?D???D???49X??o;??
=??
<t?<49X=?1<???<?t?=?o<?<???<T??=P?`<?t?<?1=D??=?7L<?/=@?=t?<???<??
<?j=,1<???=???=?w<?`B=0 ?=C?=H?9=??=L??=???=D??=0 ?=ix?=}??=?O?>M??=aG?=y?#=?C?=e`B=q??=?
==y?#=e`B=??=??=?o=y?#=???=??=???=?Q?>1'=??`>z?=???B?fB? B'??B?iB)@B	?B??BM?B
NcB?|B?GB	?WB}?B?B?RBO?B#?
A??B??B??B6B??Bf?B??BZB??B/=B?EBMBщBnB??B"ׄBZ?BW}B?fB!?OB?+B6?B@?B?BoPB?FB ??B$??B??B?`B?B??B|OB%??B?B^?B&B?,B?B??BD?B?9B*??A???BB?wBG?B?Bl;B?B??B'?RB??B8{BF?BBB>?B
?B??B?B	?!B?B??BZ?BJ?B$:AA?
?B??B??B@BJ%B?.B?DB
?QB?HB/=B??B@0B5?B??B<?B#>?Bx?B??B??B"=?B??B?[B??BAyBKiB?JBBB$?fB??B?RB?ZB?BM?B%MBB
?$BBhB<?B??B?)B?Bk?B??B*??A???B?B??B@?B?UB??A?C?+Z@??5A?5k???jA?`A?u5A??)A??]A???A?u?A?Q?A?J?A?FA?dB?@?v|A??_A?a?A?2dC?i8B	??A?ʋA{?AA?AЌ[Aj%cAd%Ac??AT?%A???&?hAQA?пA?#vA?WA??kAh??A<bF=???A?ǠAʮ]A???Aϙ?@+s>C???AU?\@?ܡA6x?@??AڰA?Z?A=$?A9Q@?O@T+A?
?A?i?@??}A?A经Bb?@??	@???C??C???ATC?/?@?1?A?|??>??A?E?A?}?A?s?A?g5A???AΕA??XA?|?A?ޗA?t?B?L@??YA???A???A?zC?h!B	ĔAҀ,A{MAB?cAЎwAjЙAe?Ae AS?A?}[?-??AP?XA??A?f?A??dÅvAh??A<?-=z??A??AA?v?A??eA?@+??C??oAVۨ@??A5@??oA??A???A={?A9 @
?N@T??A???A?mi@??YA(A??xBC?@??\@??+C?	?C???                           Q         O         :            *   	      "   5                            7            
            ,                  ?            
      7   
      )            
         	   +      1                              9         ;         )   )         !         #         %                        #                                                                                                                              #                                                %                        !                                                                                                   Nq?YN:0NlliNRqN?U?N^URNg?N?"?P??NAx?M???O?	O??O/??O?ıOM?<OXNg?3OG?N!?SN8X?Om,?O'??O?3P??N??N?5M??NK?4O??	NuE?O??O???N?M?Ou??OkI
N?1fN"a`Ow?O/?N]?Nj?AO?f?OBf<ObB?O2ކN?|?N?6WOpoN^?N???Oz+NN??Nk?3O??N??3N?	?Nk?SN??N?KO'&?N\?sO(??N??.Oy?N?zC  *  	G  ?  c  ?     ?  g  	    ?  j       "  ?  h     ?  ?  	  ?  ?  ?  ?  ?   ?  ?  <  ?  @  ?  )  ?  s  2  ?  ?  !  ?  ?  ?  "  ?  ,  ?  H  ?    d    
?  q  e  ?  b    ?  E  x  v  ?  	?  }  p  8?q????/???
?????T???e`B?o?o=o;??
;??
=@?;ě?;?`B<?/<?C?;?`B<#?
<ě?<#?
<49X<?/=o<u<?t?<?1<?C?<?t?<???<?j<??
=<j<???<?j<ě?<???<???<???=C?=8Q?=??=t?=?w=#?
='??=???=0 ?=8Q?=<j=@?=D??=P?`=T??=T??=]/=]/=u=q??=?+=?C?=???=???=?v?=?^5=?
==?
=????????????????????-*./<HJHG<0/--------`\ZVVWbgikhhkb``````????????????????????zyz|??????????zzzzzz????????????????????????????????????????????????????????????Y[at????????????th]Y?????????????????????????????NNU[gt?????????tgUON4358BN[gv{tng[MBA=:4??????


?????????????????????????????)5:AB<5/)??????

???????"!")/494/"""""""""????????????????????????????????????????????????????#")35BNPWZ[ZNB5-)#/../<HSUahnoia[UH<4/'&)/6@BOPTQOICB62-)'[eio?????????????l^[????????????????????
&%









????????????????????????
?????????!)6BKY]\XOB6)#/8<G</,#???????????????????????????


 ???????????????????????????????????????RTX^aovz?????}zna^XR????????????????????	

YV[`hqt????????tha[Yabinwz????????zynhda????????????????????????????????????????????")-)''%????~{{??????????????????
#)5:;80*#
?????????????
	???????????

???????????????????????????????????????????hkrt???????thhhhhhhh#0<>G><910#zuux}??????????????z# "#/1<AHHHG<5/-###OMKUanoqnfaUOOOOOOOO
	+6BDGJJB<6)1++16<BGGFCB:6111111?????????????????????'%????????????????????????????????????????75444;HT^abaa_XTH;77?????


???????????????#**)????516BIOQROB:655555555?????

?????????????????????????????????????????????ּӼּټ???????????D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?DӻF?S?_?l?x???????x?l?_?S?F?@?F?F?F?F?F?FčĚģĦĪĦĠĚĕčĉĊčččččččč??????????????????????????????????{ńŇňŔŠŬŠŞŔŇ?{?t?y?{?{?{?{?{?{?[?h?t?t?w?v?t?h?c?h?l?h?[?V?[?[?[?[?[?[??????????????????????????????????????)?N?t??~?u?g?[?B?5?)????????)??*?6?A?@?6?*?'????????????????????????????????ùøù???????????????????????????????????????s?i?b?c?k??????čĚĤĦĨĦĝėčā?n?[?R?N?O?[?h?tāč???????????????????????????????s?u???????????????????????????g?Z?M?D?G?R?Z?g?s???\?h?uƁƉƎƐƗƖƎƁ?u?h?^?\?Z?[?[?Y?\???????????????????x?l?`?_?^?l?q?x?????????	??"?'?"? ??	???????????????????????;?H?T?a?m?s?z?y?m?a?T?Q?H?;?9?0?.?/?3?;?/?<?<?<?9?/?)?#?!??#?#?/?/?/?/?/?/?/?/E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E????$?0?>?I?K?I?@?=?0?$????????????????????????????????????????????ѿݿ??????????????ݿѿοĿ????????ĿʿѾ[?s??????????????s?f?Z?M?E?F?<?;?A?M?[???????????????????????????????????????ҿ`?m?p?y?{?y?m?`?Y?\?`?`?`?`?`?`?`?`?`?`?;?G?K?T?U?T?G?;?.?-?.?9?;?;?;?;?;?;?;?;?G?J?T?U?U?T?G?;?.?,?.?;?;?C?G?G?G?G?G?G?ʾ׾?????	??	?	???????׾ʾ????????Ⱦ????? ?$????
?????????????ܹ????????	????????ܹѹϹ̹Ϲֹܾ????????????????׾ʾ??????????????ʾ׾?????????????????????????????????????????????????????????????????????????N?Z?g?s?u?g?Z?V?N?A?5?(??????(?5?Nìù??????????ùìåàßØààæìììì?T?`?m?m?n?m?`?T?M?M?T?T?T?T?T?T?T?T?T?T?A?M?S?Z?a?f?h?i?f?Z?M?M?A?4?/?)?.?4?=?A?ùϹܹ??ܹ۹ԹϹù??????????????????????a?n?v?r?p?z?|?z?n?m?h?a?]?U?a?a?a?a?a?aÓÝàêåà×ÓÏÌÇÄÇÈÓÓÓÓÓÓ?#?0?<?G?O?Q?M?I?@?0?#?
??????????????#ù??????????????????????????ùìæî÷ù?????ɺֺߺ????????ֺɺ?????????????????D{D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?DyDzD{?ʾ׾??????	???	?????????޾׾ʾ??žʾʻûлܻ??????????ܻл̻û????ûûûûûþ??(?4???A?D?A?A?4?/?(?"????????:?F?S?U?V?S?P?F?C?:?8?1?:?:?:?:?:?:?:?:????????
?????????????ּԼҼּ?????????????"?+?+?#???????????????????A?M?Z?f?g?g?f?^?Z?M?M?L?A?8?4?0?4?A?A?A?(?4?A?C?G?B?A?4?/?(?&?$?(?(?(?(?(?(?(?(?r?~???????????ɺԺɺ??????r?e?^?Y?\?e?r?????????????????ٺ????????????????/?2?;?=?<?;?3?/?"????"?'?/?/?/?/?/?/?	??"?$?%?"???	???	?	?	?	?	?	?	?	?	?f?r???????r?p?f?Y?Y?T?Y?b?f?f?f?f?f?f?y???????????z?y?x?v?n?m?y?y?y?y?y?y?y?y???????
??#?#? ???
??????????????????ǈǔǡǭǩǡǕǔǔǈǂǆǈǈǈǈǈǈǈǈ?@?M?S?Y?Z?Z?U?M?@?4?'?????!?,?4?;?@????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E}EuEtErEuEvE?E7ECEPEZE\EbE\EPECE7E4E0E7E7E7E7E7E7E7E7 - F ? B + k k I  4 ? - : H 4 F K ? ) D D  : 0 6 + E b _ # L ( I B ! E @ (  3 ? D R 3  . m 1 , N %  / < D Q 9 a \ b * / V C 1 G  q  i  ?  z  ?  u  L  ?  =  ^  ?  ?    ?  ?  ?  <  ~  ?  Q  \  ?  w  #  p  ?  4  !  ?    ?  D  M    ?  ?  ?  6  3  ?    ?  ?  ?  ?  ?    ?  )  ?    ?  ?  ?  z  ?  ?  y  ?  ?  c  i  ?  J  @  ?  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  *      ?  ?  ?  ?  ?  ?  p  P  2  %    
  ?  ?  ?  ?  ?  	G  	/  	  ?  ?  ?  f  7    ?  ?  i  7    ?  ?  t  ?    ?  ?  ?  ?  q  d  e  e  e  a  Y  Q  I  9  $     ?   ?   ?   ?   ?  c  X  M  A  3  %    ?  ?  ?  ?  ?  e  <    ?  e     ?   T  l  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  2  ?  ?  "  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  U  5  ?    h  P  9  $      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  g  f  f  d  `  [  U  M  F  :  -      ?  ?  ?  ?  q  N  ,  ?    ?     i  ?  ?      ?  ?  ?  u  !  ?  ]  ?  0  @  ?                  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?        !  "  !      ?  ?  x  O  &    ?  ?  ?  ?  ?  :  y  ?  ?    "  ?  [  j  [  ;    ?  <  ?    ?   ?       ?  ?  ?  ?  ?  ?  ~  t  \  >    ?  ?  ?  u  6  ?  H            	    ?  ?  ?  ?  ?  ?  ?  d  3  ?  ?  w  H  ?  (  ?  ?  ?      !      ?  ?  ?  9  ?  `  ?    c    n  l  q  s  |  x  r  t  ?  ?  e  ?    ?  ?  u  H    ?  :  h  c  \  S  G  9  (      ?  ?  ?  ?  ?  ?  ?  g  A  ?  *     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  4  h  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  /  ?  ?  I  ?  ?  ?  ?  ?  ?  ?    e  :    ?  ?  ?  Y  .    ?  ?  ?  ?  h  F  	  
  
    ?  ?  ?  ?  ?  ?  ?  |  a  D  &    ?  ?  ?    ?    -  U  u  ?  ?  ?  ?  ?  ?  p  W  5  ?  ?  J  ?  R  A  ?  ?  /  W  v  ?  ?  ?  ?  u  e  P  2    ?  3  ?  ?  ?   ?  ?  ?  ?  ?  ~  ~  |  z  y  u  i  V  >  $    ?  ?  |  9   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    K    ?  ?  o  ?  Y  L  i  ?  ?  ?  ?  ?  ?  ?    k  T  <    ?  ?  ?  m  -  ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   u   j   _   U   J  ?  ?  ?  ?  ?  ?    	        '  /  7  >  F  N  V  ]  e  <  1  '         ?   ?   ?   ?   ?   ?   ?   ?   ?      j   V   A   -  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  d  F  #  ?  ?  ?  >  ?  ?  -  @  <  8  4  /  &        ?  ?  ?  ?  ?  ?  h  K  /     ?  q  ?  ?    I  k  ?  ?  ?  ?  ?  ~  F  ?  v  ?  d  ?  ,  ?  !       $  (  $      ?  ?  ?  ?  ?  \  :  $  ?  ?  h    ?  ?  ?  x  n  c  X  M  B  6  *        ?  ?  ?  ?  ?  ?  s  r  k  _  O  ;     ?  ?  ?  ?  o  O  .    ?  ?  K  ?  &  2  '      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  j  X  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  8  ?  ?  a    ?  f  ?  [  ?  ?  ?  ?  ?  x  m  _  P  A  1  !      ?  ?  ?  ?  d    ?        !         ?  ?  ?  ?  ?  p  D    ?  ?    ?    {  ?  ?  ?  ?  ?  ?  ?  j  M  (  ?  ?  c  ?  t  ?  ?  ?  ?  U  j  ~  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  i  Q  9    ?  ?  ?  ?  ?  ?  ?  ?  p  \  G  1       "  !        ?  ?  ?  ?  w  ?  ?  ?  m  :    ?  ?  d  ?  ?  ?  ?  ?  ?  ?  ?  n  O  '  ?  ?  ?  X  !  ?  2  ?  ?  +  (        ?  ?  ?  ?  ?  m  F    ?  ?  L     ?  4  K  ?    ?  ?  )  ?  ?  ?  ?  ?  ?  8  ?  )  ,  ?  ?  ?  ?  K  ?  E  E  >  0      ?  ?  ?  k  3  ?  ?  P  ?  ?  8  ?  ?  f  ?  ?  ?  ?  ?  ?  ~  [  2    ?  ?  X    ?  ?  N    ?  ?        	  ?  ?  ?  ?  ?  f  8    ?  ?  3  ?  ?      d  \  U  L  D  +    ?  ?  ?  ?  R    ?  ?  C  ?  ?  ,   ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  [  C    ?  ?  %   ?  
?  
?  
g  
M  
;  
*  
  
  	?  	?  	?  	L  ?  j  ?  \  ?  ;  ?  ?  q  j  b  V  I  ;  +      ?  ?  ?  ?  ?  f  ?    ?  ?  e  e  \  S  J  A  7  *        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  e  I  (     ?  ?  W  ?  ?  ?  ?  :  b  X  L  <  +      ?  ?  ?  ?  o    ?  ?  D    ?  ?  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  V  @  )         ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    u  j  `  V  L  B  E  C  B  ;  4  '    
  ?  ?  ?  ?  ?  r  i  a  [  V  O  H  x  v  s  q  n  g  a  Z  P  ?  .      ?  ?  ?  ?  ?  ?  q  ?  1  `  s  t  k  a  T  D  1    ?  ?  ?  +  ?     W  x  ?  ?  ?  z  c  O  :      ?  ?  ?  ?  m  M  ,    ?  ?  T  ?  	?  	?  	?  	?  	?  	?  	|  	\  	2  ?  ?  F  ?  b  ?      ?  r  
  }  ?  ?  ?  ?  t  Z  ?  #    ?  ?  a  $  ?  ?  n  .  ?  ?  
?  9  _  o  i  W  7  
  
?  
?  
H  	?  	w  ?    /    ?  T  /  8  &    ?  ?  ?  ?  x  W  @  8  2  )         ?  ?    n