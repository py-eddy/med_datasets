CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����E�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��v   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�-      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��
=q     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v�33334     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >["�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B.��      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�VQ   max       B/$s      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?9ދ   max       C��      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?K�   max       C��       �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��v   max       P���      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�{���m]   max       ?ۜ�ߤ@      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >%      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��
=q     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v�
=p��     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�u@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�j~�   max       ?ۚkP��|     �  QX                     &      
         	      =               ^   �   )   L      !   4      	   =   2   B            +            0      %   \   <   G               #                     
      O$֨OeY�N�V!M��OkO�NC�Oѳ�N,��N��NȺ�N��/O��O�'�P�^EPyNzWO"O�/P9��P��7O�#hP��N"$O�O�A�NJ�N���P!�O���P@�
Nu�N6�NK�P-O�n�N�/oOw
�O��N�h�O���P�cO�M�PtvN���Nn�O�jhO�X6Op�M�tO���M��vOs� O��lN(buN�uN��OE�����T����o;o;��
;��
;ě�<o<#�
<T��<u<u<�o<�o<�o<�C�<���<��
<�1<�1<�1<�j<�j<���<�`B<�`B<�h<�<�<��=+=+=C�=\)=�P=�P=��=,1=,1=<j=<j=<j=@�=H�9=L��=P�`=T��=e`B=e`B=ix�=ix�=m�h=q��=q��=�t�=�t�=�-0*,.167@BOPPTUWUOB60�~������������������&))5765-)Z[eggmhllgb[ZZZZZZZZSR[it�����������tg[S��������������������WW\ciz�����������znW}�����������}}}}}}}}�����������������������������������������������������������������
#(#
�����IIT]demz������zm^VTI����5Ng{{ti`C)����KM[t�����������m[SSK�����������������������������
���ea`bbh������������te����)5>>2.52)���)5N���������[B,#!#&/<HUanwxnaU</,##��5DPV[ji`L5���
 
����������������!*6BO[acb[TOB6+��������������������~y�������������~~~~����
#/6@PF9/#���������������������mv{��������������znm�utlst��������������..5BHNONB5..........��������������������(!**)5Ng������t[N5(��
#/78961/
����
##$$##
����-,+)/5;HTX\^]\WTH;/-�������#.*#���������

����������$'&"����������)-+.1BB)��EO`h���������nhc`WE��������������������~yv���������������~~"./;?;70/" ����)0:GIFB6, -6BGIIF@=2)��#8<HSUTQJH<#
�� $�������������������������


��������������������������������� ���������������������������� �
#$#!
yzz���������{zyyyyyy#/<HJIJJHA<90(#�l�y���������������������������y�m�d�a�lĚĦĳĿ��������ĿĦĕčā�t�s�yāčĒĚ�G�T�`�i�m�m�m�d�`�T�R�G�A�>�G�G�G�G�G�Gìñù������ùìàÜàëìììììììì���������������������������������������������ʾ˾̾ʾ�������������������������������5�A�R�Z�f�a�Z�N�5�(�����������ÇËÌÇ�z�n�a�_�a�n�zÄÇÇÇÇÇÇÇÇ�����������������������������������������f�s�|���������s�f�`�]�`�b�f�f�f�f�f�f�#�$�/�5�<�@�>�<�3�/�#���
��#�#�#�#�#����������
�� ��������������Z�g�o�������������s�g�N�A�4�)�.�A�N�R�Z����,�/�-�5�@�=�(����ݿѿ��������Ŀѿ������������������������s�g�Q�T�N�G�Z�g�����������������������������������������Ҿ��	��"�.�;�A�G�[�S�G�;�.�"���	��������(�4�A�M�T�K�L�Q�S�M�4�/�(��������0�<�G�U�]�j�m�b�U�<�#����Ĭĵ��������6�O�hĚĦĨğą�s�[�B�6�&�������6����	�������	�������������������Ƴ����$�=�I�G�=�$������ƧƁ�e�V�hƎƳ���������������������������������������˻����������ܻ������������ûлܻ�龥�����ʾ׾���ؾʾ����|�s�p�u���������Ľнݽ����������ݽ۽нĽ��ĽĽĽļ����������������������������������������;�T�a�m�x�y�o�a�;�	�����������������	�;���.�7�9�3�'�����Ϲù��������Źܹ�Š������������������ŹŭŋŃ�~�x�|�|ŇŠF=F=F=FJFVFXFcFcFdFcFVFJFHF=F=F=F=F=F=F=�(�4�;�9�4�)�(�'� �#�(�(�(�(�(�(�(�(�(�(�����ʼʼμʼ��������������������������������	�;�T�f�j�g�\�G�;�/��	����������������	��&�.�:�-��	������;ʾ��Ⱦ�y�{���������z�y�m�`�T�M�K�T�X�`�m�u�y�y�#�0�<�I�R�U�V�]�Q�I�<�0�#��
������ �#�)�5�N�[�e�g�c�[�U�O�N�B�4�)���
���)�A�M�Z�f�p�s�����s�f�Z�M�J�A�=�A�A�A�A�����
����������²¦¦²»¾������f�r�������������������r�M�@�1�3�:�M�f�@�A�M�X�B�4�������ϻ˻лܻ������@�n�zàù����������ìàÓÇ�z�f�^�\�e�c�n�������������������������~�y�s�{�~�������T�a�e�m�r�z�z�~�z�w�m�a�`�W�T�Q�T�T�T�T�����-�:�D�B�:�'�!�������غ����e�r�~���������ɺѺҺɺ������r�Y�L�@�P�eD�D�EEEE*E.E2E*EEED�D�D�D�D�D�D�D�T�U�[�`�b�`�T�L�G�F�G�R�T�T�T�T�T�T�T�T�4�M�Z�s�v�z�s�f�^�Z�M�A�4�(�����(�4�M�Z�\�Z�R�M�A�A�?�>�A�L�M�M�M�M�M�M�M�M�����������������������������������������������Ľͽ˽ɽý������������������������"�.�;�<�=�;�.�"���"�"�"�"�"�"�"�"�"�"ǔǡǭǸǶǭǦǡǘǔǏǓǔǔǔǔǔǔǔǔ�r�������|�r�f�Z�Y�X�Y�f�j�r�r�r�r�r�r��-�.�"� �������������������������� M G ? | D E A j ( 5 N h L 9 " 7 b R S < D J P ? + � � L W 2 P N I / M > Q * l J 7 Q $ B s J ` m n & � _ R ; 5 < U    x    �  7  �  L  �  �    �  �  �  U  �  �  �  h  �  �  �  %  �  6  0  �  �  �  S  �  G  y  X  .  
  9  �  �  }  �  �  �    �    �  s  �  a  <  {  9  $  ?  O  �  �  ɼu%@  %@  ;ě�<�h<o=49X<D��<���<�`B<�j<�j=8Q�=���=0 �<�`B<��=t�=�>["�=y�#=���<�/=q��=���=+=�P=�^5=��
=ě�=<j=�P=�w=��-=u=L��=�o=�Q�=H�9=���>I�=��=��=�O�=ix�=��-=��T=�^5=q��=��w=q��=���=��=�%=��=�1=���B+�B>�B��B	DB	�B ��B�B�B�BFB�KB?�A�AB�B
JIB�	B`BGB�vB��B�CB�AB��B#"�BH	B"ծB�hB�*B�B�OB��B��B"|�B`<BWsB A�^�B��B#��B1B�Bz�B[xBi;A��B�IB�GB�CB.��Bc�BB�B��B,]�B-YSB�B�.BQ�B>�B ��B�B	?eB	��B �UB��BAzB7}B7B��B�@A�n�B��B
�nB�OB@�BV�BŲB��B��B�AB�B#?^B@&B"�jB��B@BBLB3�B��B��B"�BQ�B�.B$�A��	B%�B#��B?nB�B��B�B@A�VQBAhB>�BB�B/$sBB�B6�B��B,{B-FB(�B��BHKA��A߲bAgE�A��A���AN A���A�.NAr�ACeA��7A�S�A���A�{�A�oA��<A`ۑA8�GA�'UA��A҄�B�A�vI@���AL��A+�?@�ƄA���?9ދA�6�C��A7�&@�V
A���AX�Aj�A�_{A���A?�/A��]@��<@��{A���@9A��9@_��@�YC�]jAg�/A<K�A=<IA�;A"�&A`@RB��@��A�'A��A��hAg�zAˮ�A��AMF�A�~pA��pAr�(AC�A�pJA��A��A��A�j?A�{+Aa �A6�A�$�AفiAҀ>B��A��J@�,�ALԃA*�%@���A�8e?K�A�t:C�� A7 �@�>�A��AZx`Aj��A��A��AA>�A� �@� @��A�}f@�+A�~�@c�
?��vC�W�AgA;�A<�-A���A"5GA_ߐB�5@�rA��                     '      
      	   	      =               _   �   )   M      "   4      	   >   3   B            +            1      &   \   =   H               $                                                                     9   '            /   5      =         !         1   #   +            +                     '   '   %                                                                                    5               !         7                  %      '            %                     !                                                OxqN�4KN�V!M��O:/�NC�O���N,��N��NF�NUj�O��O̼P���O�%�NzWN�#�OqX"O��O��N�KZPx7N"$O\��OC�NJ�N���O�@�O���PօN^��N6�NK�P�ZO�n�N�/oO�'OrϖN�h�OY�O�uyOo�Oۻ�N���Nn�O��3O�VnOp�M�tO�LSM��vO$jO��lN(buN�uN��OE��  �  W  �  �  �  i  �  �  �  �  �  �  .  >  �  �  �  �  �  �  z  �  �  �  �        �  �  �  �        �  2  �  �    �  }  	�  \  �  '  �  	t  �  �    U  L  ?  �  X  ���h�o��o;o<t�;��
<D��<o<#�
<���<�o<u<�/<�9X<���<�C�<��
<�j=Y�>%=#�
=C�<�j<�=8Q�<�`B<�h=,1=#�
='�=C�=+=C�=��=�P=�P=<j=T��=,1=e`B=�O�=�o=y�#=H�9=L��=T��=aG�=e`B=e`B=q��=ix�=�%=q��=q��=�t�=�t�=�-1,./36<BLOQSUSOJB611��������������������&))5765-)Z[eggmhllgb[ZZZZZZZZV[^lt����������tg\[V��������������������_cdgnz�����������zi_}�����������}}}}}}}}�����������������������������������������������������������������
#(#
�����]^adkmxz�����zymgaa]�����5Ngxqf[>)��XUT[gt����������tg[X��������������������������	

�������fght�������������uif����"$'**&����30/25BN[gx~~ymg[NB:3-/0<HPUZUQH<0/------���(5@GNV`aTB)���
 
���������������')/6<BOWZ[[YTOB66/*'��������������������~y�������������~~~~���
#/6?DC;/#
�����������������������t}����������������ztmtt��������tmmmmmmmm..5BHNONB5..........��������������������"!,-2BNg�����tg[N5)"��
#/78961/
����
##$$##
����:6338;HTUYYWTRIHE;::������������������

������������ ���������$*22,����\[dht����������tlji\��������������������~yv���������������~~"./;?;70/"����)/9FHEB6/)6BEGGEB;/)��#8<HSUTQJH<#
�� $�������������������������


�������������������������������� ���������������������������� �
#$#!
yzz���������{zyyyyyy#/<HJIJJHA<90(#�l�y�����������������������y�p�l�g�d�l�lĚĦĳĵĽĳįĦĚčā�āĄčĐĚĚĚĚ�G�T�`�i�m�m�m�d�`�T�R�G�A�>�G�G�G�G�G�Gìñù������ùìàÜàëìììììììì���������������������������������������������ʾ˾̾ʾ������������������������������(�5�A�L�Z�^�Y�N�A�5�(��� ������ÇËÌÇ�z�n�a�_�a�n�zÄÇÇÇÇÇÇÇÇ�����������������������������������������f�s�������v�s�o�f�d�f�f�f�f�f�f�f�f�f�<�>�<�<�1�/�#�����#�/�8�<�<�<�<�<�<����������
�� ��������������Z�g�l�s��������s�g�Z�N�A�5�5�:�A�N�O�Z�ѿ���&�)�)�'�/�7�3����ݿѿ������������s�����������������������m�e�a�a�`�c�l�s���������������������������������������ҿ.�;�>�G�T�W�P�G�;�.�"���	��	��"�#�.�4�A�E�M�I�H�I�M�L�A�4�/�(� ������4��#�0�7�>�?�:�0�#���������������������B�O�[�h�n�x�{�y�t�h�[�O�B�6�2�,�+�.�6�B������
�����������������������������ƧƳ�����0�=�;�0�$�������ƚ�}�h�lƎƧ���������������������������������������˻лܻ������������ܻû����������ûо����ʾӾ׾پ׾ʾ������������������������Ľнݽ����������ݽ۽нĽ��ĽĽĽļ����������������������������������������U�a�k�r�u�s�f�T�/��	����������"�;�H�U����#�'�1�0�'�������ܹ¹��ùϹܺŠ������������������ŹŲŔŉŃŀŁňŌŠFJFVFWFbFcFcFcFVFJFIF>F?FJFJFJFJFJFJFJFJ�(�4�;�9�4�)�(�'� �#�(�(�(�(�(�(�(�(�(�(�����ʼʼμʼ������������������������������	�;�T�a�e�c�Z�C�/��	���������������������	��&�.�:�-��	������;ʾ��Ⱦ�y�{���������z�y�m�`�T�M�K�T�X�`�m�u�y�y�
��#�0�<�A�E�?�<�0�#���
��������
�
�)�5�B�N�[�^�c�^�[�N�G�B�5�)������)�A�M�Z�f�p�s�����s�f�Z�M�J�A�=�A�A�A�A����������
������������������������ؼr����������������r�Y�M�@�;�=�D�M�Y�f�r�'�4�@�C�C�:�4�'��������ܻ�����'Óàù��������üìàÓÇ�z�p�f�d�d�n�zÓ�������������������������~�y�s�{�~�������T�a�e�m�r�z�z�~�z�w�m�a�`�W�T�Q�T�T�T�T�����-�:�C�A�:�&�!�������ܺ����e�r�~���������ºʺκɺ��������Y�L�D�T�eD�D�EEEE*E.E2E*EEED�D�D�D�D�D�D�D�T�U�[�`�b�`�T�L�G�F�G�R�T�T�T�T�T�T�T�T�A�M�Z�f�s�w�s�f�Z�M�A�4�(�����(�4�A�M�Z�\�Z�R�M�A�A�?�>�A�L�M�M�M�M�M�M�M�M�����������������������������������������������Ľͽ˽ɽý������������������������"�.�;�<�=�;�.�"���"�"�"�"�"�"�"�"�"�"ǔǡǭǸǶǭǦǡǘǔǏǓǔǔǔǔǔǔǔǔ�r�������|�r�f�Z�Y�X�Y�f�j�r�r�r�r�r�r��-�.�"� �������������������������� : ? ? | E E 9 j ( 6 ] h F : 0 7 T K >  * I P ? & � � F N / M N I + M > D  l 2 5 O   B s K c m n ' � O R ; 5 < U    -     �  7  �  L  u  �    U  �  �  O  :  {  �    
  �  -  �  )  6  �  �  �  �  5  2  �  �  X  .  �  9  �  4  �  �  �  �    �    �  c  �  a  <  :  9  �  ?  O  �  �  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  �  �  �  }  m  Z  B  )    �  �  �  �  �  �        *  4  @  L  T  V  U  O  D  7  %    �  �  �  e  9  �  �  �  �  �  �  �  �  �  �  z  d  N  7  !     �   �   �   �  �    8  Y  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  j  K  )     �  �  �  %  �    �  i  f  b  ^  [  W  S  P  N  K  H  E  C  ?  8  1  *  $        �  �  �  �  �  �  p  T  *    �  �  ;  �  �  B  �     �  �  �  �  �  �  �  }  r  f  Z  M  ?  1  $       �  �  �  �  �  �  �  |  v  n  f  Z  L  <  +      �  �  �  �  �  h  >      3  F  Y  m  z  �  �  |  n  `  N  8    �  �  x  D  G  �  �  �  �  �  �  �  �  �  �  |  v  q  k  f  a  \  U  K  A  �  �  �  �  �  �  �  �  �  s  f  Z  M  7  "  	  �  �  �  �    ]  �  �  �    )  .  +    �  �  �  �  F    �  u  9  �    3  >  6  %    �  �  |  A  	  �  �  �  �  �  W  �  2  �  9  Y  o  �  �  �  �  �  �  �  ~  f  E    �  �  A    �   �  �  �  �  �  u  d  Q  =  !    �  �  �  �  �  �  �  q  a  P  �  �  �  �  �  �  �  �  t  f  W  H  $  �  �  �  P     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  &  �  �  �  \  	  	�  
#  
�  )  m  �  �  �  \  *  
�  
�  
#  	�  �  �  X  �        �  E  �  y  2  �  b  �  �  �  �  3  [  #  �  
�  �    y  �  �    ;  V  i  s  v  y  t  Z  5    �  o    �    �  d    �  �  �  _  2    =  $  �  �  q  /  �      >  �  �  �  �  �  �  �  }  y  v  s  o  k  e  _  Y  T  +  �  �  �  Z  �  �  �  �  �  �  �  �  |  X  )  �  �    !  �  w  1  �  �     I  r  �  �  �  �  �  �  �  w  F    �  �    m  x  O   �              �  �  �  �  �  �  �  v  ]  E  +     �   �      �  �  �  �  z  ^  D  *    �  �  �    F    �  c    8  �  �  	  
  �  �  �  �  4  �  �  L    �  @  �  7  @  �     c    �  �  �  v  r  h  Y  ?    �  �  L  �  L  ]  	  �  u  �  �  �  �  �  T    �  W  
  �  �  =  �  �    e  X  �  �  O  b  /    �  �  �  �  �  c  9  >  v  <  �  �  N  :  �  �  �  �  �  �  �  �  �  �  �  �  n  P  3     �   �   �   �   x    �  �  �  �  �  �  �  �  �  �  �  u  m  d  [  _  f  n  u  �  
  	  �  �  �  �  �  Y  #  �  �  �  [    �  J  �  e         �  �  �  z  W  6    �  �  �  {  D    �  �  T    �  �  �  �  �  �  h  P  8      �  �  �  �  l  K  (    �  �  �  �  �    $  /  2  +      �  �  X  �  �    �    �  �  $  �  �  �  �  �  �  �  �  x  P    �  p  
  �     x  �  �  �  �  �  �  �  �  �  x  c  l  z  �  w  j  b  b  a  G  #  �  �  �            	  �  �  �  z  H    �  �    �  n  }    _  �  �  �  �  �  �  f  !  
�  
`  	�  	X  �  �  �  o  P  3  �  '  H  ]  o  y  {  o  I    �  �  H  �  |  �  <  W  |  '  	f  	�  	�  	�  	�  	�  	�  	P  	  �  6  �  Z    �  /  �  �  �  �  \  K  0      �  �  �  �  �  �  �  �  j  =    �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  v  e  N  0    �  �  �  &  &  !    �  �  �  �  \  /  �  �  �  �  e  "  �  :  �   �  �  �  �  �  �  �  ]  z  r  P  (  �  �  T  �  f  �  E  �  R  	t  	N  	&  �  �  �  `    �  �  ^    �  .  �    w  �  �  c  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  X  A  *     �  �  �  �  �  �  �  �  t  Y  =    �  �  �  ~  Q  #  �  �      	    �  �  �  �  �  �  �  �  �  �  �           !  )    !  4  L  S  I  7    �  �  �  W    �  �  @  �  D  �  �  L  <  .  "        �  �  �  �  �  s  T  2    �  �  n  3  ?  :  6  2  -  )  #          	    �  �  �  �  �  �  �  �  �  �  �  _  @     �  �  �  }  H    �  �  �  V  $  �  t  X  >  $    �  �  �  �  i  C    �  �  x  .  �  �  =  �  �  �  �  �  �  t  N  '    �  �  �  �  g  O  6       
  1  u