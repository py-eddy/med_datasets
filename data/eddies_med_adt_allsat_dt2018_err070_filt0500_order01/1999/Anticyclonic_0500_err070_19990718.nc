CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�V�t�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�ָ   max       Pԭn      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       =�{      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E���R     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vN�Q�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�"           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >��+      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�0?   max       B,t�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�Mp   max       B,N�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =#d�   max       C�W+      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @��   max       C���      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�ָ   max       P�/�      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�T`�d��   max       ?�'RT`�e      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       > Ĝ      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E�          	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vN�Q�     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @P�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�u�          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?�'RT`�e        R<                     y      )         @   0   
         C   	   	            C   6   	      -               4            "            2         /      	         t      /   �   !   k            
   Oi�N	�N-S
N��5N�,�O3;�PԭnN=�eO���N��@Ne�PHw�P��O��N@^�N)��P���O4N���N�0N���O���O�nP
�N QN0rKO�NO5AKNO��N3{�NHN�O��ANs?GO)�~N�m�O�OuO�)�N�O�N.:�P	��N���N���P�nN��
N
��NF�O ��O�\�O� 0O�<�P9-�O2pRO�e�N߸qM�ָN:\�N<�N"뜼T���t��o��o%   :�o;��
;��
<o<#�
<49X<T��<e`B<���<���<��
<�1<�9X<�9X<�9X<�j<�j<���<�/<�<�<��=o=o=o=+=\)=\)=�P=�w='�=,1=,1=0 �=8Q�=8Q�=<j=@�=@�=D��=H�9=T��=e`B=e`B=e`B=m�h=}�=}�=}�=�%=�O�=�\)=�{����������������������������������������;:<AHQTSH<;;;;;;;;;;	"/962//" 	��������������������Yainoz��������zqncaY�������)NbgeN;��������������������������
#/<DHOVYS</#�")-6BOP[OMBB6)��������������������[\gt������������r`\[�������
����������������������������xx|�������xxxxxxxxxx #%/291/,#        �����5JV]_\R5)������������������������!#$/:<=HIHF<;/%#!!	
"#,0;610*#
		����������������������������

������
#/<HSSUSH</#�����)BT[fg[NB)������� ��������������������������������
)393)
�����������!(**,)&��4/*.5:BEEBB544444444������������������������������������������������������������z|z�����������zzzzzz���)-5852)���_bgnt{���������{nb__����)6@BB=6)�������������������������������������������������������������!"/Gm|��{maH;/%+#%!>=<BBMOSV[\[ZUOB>>>>�����������������5BGFIG?5)��LNPW[goqtttg_[RNLLLL##$+05<><70#########,+/;<?AB?<:50/,,,,,,������
$'(#
����������$+-"����536BO[huxxvuh[OGB>95�������	���������98BN[g���������tgNC9z{�����������������z���������	���������������������������������������������VX]afmppmaVVVVVVVVVVa_\]ajnsvnnaaaaaaaaaNMUannndaUNNNNNNNNNN�������������Ǻ������������������������������ʼμռʼ���������������������������D�EE
EEEED�D�D�D�D�D�D�D�D�D�D�D�D��T�Y�a�d�h�k�a�T�H�;�9�9�;�A�H�P�T�T�T�T�C�O�R�X�\�c�d�\�\�Y�O�C�6�6�4�,�6�9�C�C�������������������������������������������
�<�n�y�{�u�p�e�S�<�0�����ķĲľ����¿��������¿¿²ª­²¼¿¿¿¿¿¿¿¿������������!����������������޾������ʾԾ׾۾޾׾׾ʾ������������������������ùϹϹϹù������������������������H�a�m�t�u�q�i�\�H�/����������������/�H���ûͻλʻ����������x�F�C�_�x�����������z�������������������z�n�m�d�f�m�o�u�z�z���ûлٻֻлû��������������������������"�/�;�@�;�3�/�"�����"�"�"�"�"�"�"�"������/�0�$�����ƳƚƎ�u�F�D�K�T�\�u�̾M�Z�c�f�l�s�������������s�f�Z�X�N�J�M�z�����������������������������{�z�w�z�z�����������������߼�����4�A�M�T�M�L�M�Q�M�A�4�(�'�����(�)�4�A�M�R�Z�f�n�x�x�o�f�Z�M�A�4����(�9�A�m�������������������y�m�Y�O�K�I�K�T�`�m���������� ���������������������������h�t�āċā�t�h�d�c�h�h�h�h�h�h�h�h�h�h�.�;�G�T�`�j�`�U�T�G�;�:�.�-�.�.�.�.�.�.�5�N�h�j�e�d�[�N�B�5�)������	��)�5�H�U�a�n�o�p�n�f�a�U�H�<�/�*�/�:�3�8�<�H���
��#�)�#���
�������������������������(�(�/�(���������������@�M�Y�^�f�k�f�Y�M�D�@�9�@�@�@�@�@�@�@�@����	��=�H�I�A�5�(��������������G�T�`�f�e�`�W�T�Q�G�;�3�;�?�G�G�G�G�G�G�f�g�n�p�o�l�i�f�^�Z�S�M�B�=�A�B�H�L�Z�f��� �����!�"�!�����������������~���������������������r�b�M�M�R�V�^�n�~�"�T�a�z��j�a�T�;�/�!����������	��"���������ĽԽн̽Ľ�����������z����������������������������������
�#�:�8���������ĿıĪĨĭĹĿ�������
�ܻ���������������ܻػջܻܻܻܻ�!�-�:�;�F�F�Q�F�:�-�!� ��������������ɿϿϿֿܿѿ��y�`�T�M�J�N�V�`�y�����������������ݿܿݿ�����'�*�4�@�I�@�:�4�/�'�!�'�'�'�'�'�'�'�'�'ǔǡǫǡǠǔǈ�{�o�h�o�{ǈǈǔǔǔǔǔǔ�������ʾ׾ؾ־ʾǾ����������������������������ʼ��������ּʼ���������������������������������������r�[�T�[�f�r������������������������y�l�`�X�R�P�S�d�`�l����)�E�Q�V�W�Q�L�B�6���������������������������
�������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|DD�D�D�ÇÓØàìíìëæàÓÇ�|�z�u�w�z�}ÇÇ�a�l�n�n�n�n�a�`�\�]�a�a�a�a�a�a�a�a�a�aŭŹ��������Źŭŧţŭŭŭŭŭŭŭŭŭŭ�������û̻û������������������������������!�$���������������� F Y 7 @ E . " L < M } 4 k V ) Y 8 F M . 4     J C l - F D R U < . Y f 0 B a E [ K J 3 8 = � V K - G # 1 <  r > N E  a  6  J    3  �  �  b  k  :  z  w  �  @  W  D  w  :  �  �    r  �  �  ?  �    �  m  P    �  z  �       '  (  Q  �  �  �  �  �  3  �  �  |  �  �    |  /  �  K  T  e  X;ě���`B%   ;D��<#�
;�`B>%<���=H�9<��
<e`B=���=y�#<�h=+<�j=�9X<��<�<�=\)=@�=�j=��
=�w=o=���=H�9=,1=��=�P=� �=�w=]/=]/=���=�t�=��=8Q�=��=q��=ix�=��=P�`=ix�=T��=�+>/�=� �=��>��+=��>,1=��T=�+=�t�=��
=�9XB"&�B"r�B1�A�0?B'�Bv�Bo�B ^�BF�B�B:�B
�zB#BARB�NB�B�BX�B[�B%�B��B#N�B�$B��BB1�B9(B�RB["B��BK�B�BB�B(�uBByB�B"��BMgA��B�MB(�Bp*B�B%��B��B��B��BNB,t�B	��B*[BǯB!��B�A��Bs�B
B">�B"�RB?�A�MpBy�B�B;B >�B��B�2B>�B
z�B#8�B�B��B5B?OBP�BD�B%>�B��B#�4B>�B�TB6�BC�B=%B�TBA�B��B��B�B4wB�5B(��B��BʥB"�UBo0A���B�B�B@B�QB%�B�	B�sB��B@!B,N�B	�7B;�B��B!�qBC�A���B�EBѐ@p@��aC�W+A�j`BIA���A�s�A�U�A��^AO�=#d�A�v�@��:A��G@���A�'pBD�ABemA�^SA fA9ËA;ڰAlHmA�ίA܇
Ae�A��kA�pHA�
�A�_�@�h|A��AfX�A?c#@[��@sA��A"qZA��A��5@���@q��Apd�A�#m@͕�B�ALo#@��Y@�A�A���Aщ+C���A�5�A�ҚA�KC@�LIA4	�@��@��C�W�A��#Bp�A�p~A�A�x�A҃�AO}nC���A�@���A���@���A���BF�AB��A�|�AI^A9 �A<�QAl��A��MA܇�Acb�A��AŊEA�n�A��=@�@�A�q�Af��A>�d@[�@��A��A#"NA�"A䃀@�J"@lW�Ap�"A�@��B��ANʲ@�^�@䡘ATA�KeAыlC��0AɬxAƀqA�u�@��A4v         	            z      *         @   0            D   
   	   	         D   6   
      .               4            #            2         0      
         t      0   �   !   k                                    ?               /   +            ;               !      '         %                           #   %         '         %               #      #   )                                          )                              9                     !         !                           !                                                                  N�+N	�N-S
N��5N�RO3;�P)��N=�eO$k*N��@Ne�OɝSO��O��N'JHN)��P�/�O4N���N�0N��>O�o�Ou�NO�>rN QN0rKO�3�O]~NO��N3{�NHN�Oe�0Ns?GO)�~N�m�O�-	O��ON�O�N.:�O?��Na�NnI�O�&�N��
N
��NF�O ��O��XOxj�ONsO�L:OhUN�_N߸qM�ָN:\�N<�N"�  B  �  �  i  \  f  &  n  g  �  >    �  �  �    �  �  �  Y  �  �  	.  D    �  �  Z  f  r  q  s  %  �  �  ;  �  �  I  G  *  �  d    |  k  �  �  �  �  �  =  [  �  J  �  �  �ě��t��o��o:�o:�o=e`B;��
<�9X<#�
<49X=��=�P<���<��
<��
<���<�9X<�9X<�9X<�/<ě�='�=\)<�<�=C�=C�=o=o=+=8Q�=\)=�P=�w=49X=8Q�=,1=0 �=�+=D��=L��=u=@�=D��=H�9=T��=���=ix�=�t�> Ĝ=��=�"�=}�=�%=�O�=�\)=�{����������������������������������������;:<AHQTSH<;;;;;;;;;;	"/962//" 	��������������������Yainoz��������zqncaY������/<B>5)������������������������#/<EHMPHH<3/'#")-6BOP[OMBB6)��������������������fiiot������������tjf�������

����������������������������yy}�������yyyyyyyyyy #%/291/,#        �����5GSZ[O5)������������������������!#$/:<=HIHF<;/%#!!	
"#,0;610*#
		���������������������������

������
#/<?HJLE</#
���)5BO\`_[NB)�������� ����������������������������������
%-1/%
���������%)))��4/*.5:BEEBB544444444������������������������������������������������������������z|z�����������zzzzzz���)-5852)���_bgnt{���������{nb__����)6=@?<6)��������������������������������������������������������������9:;?HNTantsmjaTHD?<9??=BGOSYXROB????????�������������������)5<>>?>5)LNPW[goqtttg_[RNLLLL##$+05<><70#########,+/;<?AB?<:50/,,,,,,������
$'(#
���������������B>:66BO[htwwuunh[OHB��������������������MIHJQ[gt��������tg[M�~�������������������������� ������������������������������������������������VX]afmppmaVVVVVVVVVVa_\]ajnsvnnaaaaaaaaaNMUannndaUNNNNNNNNNN���������������������������������������������ʼμռʼ���������������������������D�EE
EEEED�D�D�D�D�D�D�D�D�D�D�D�D��T�Y�a�d�h�k�a�T�H�;�9�9�;�A�H�P�T�T�T�T�C�O�P�W�\�b�c�\�O�C�9�6�6�.�6�@�C�C�C�C��������������������������������������������0�<�K�S�T�J�<�0�������������������¿��������¿¿²ª­²¼¿¿¿¿¿¿¿¿��������������������������������������ʾԾ׾۾޾׾׾ʾ������������������������ùϹϹϹù������������������������/�;�H�T�Y�]�Z�Q�H�/�"�	�������������/���������������������x�m�k�l�o�x��������z�������������������z�n�m�d�f�m�o�u�z�z���ûлֻԻлû��������������������������"�/�;�@�;�3�/�"�����"�"�"�"�"�"�"�"���������������Ƴƚ�u�L�I�N�X�hƎ�̾M�Z�c�f�l�s�������������s�f�Z�X�N�J�M�z�����������������������������{�z�w�z�z�����������������߼�����A�G�G�F�E�A�4�/�(�!�!�(�4�9�A�A�A�A�A�A�A�O�Z�f�n�w�w�n�f�Z�M�A�4�����(�<�A�`�m�y�����������������y�`�V�R�Q�R�T�]�`��������� �����������������������������h�t�āċā�t�h�d�c�h�h�h�h�h�h�h�h�h�h�.�;�G�T�`�j�`�U�T�G�;�:�.�-�.�.�.�.�.�.��)�5�N�d�h�c�b�[�N�B�5�)��������H�U�a�l�n�o�m�d�a�U�J�H�E�>�<�6�:�<�F�H���
��#�)�#���
�������������������������(�(�/�(���������������@�M�Y�^�f�k�f�Y�M�D�@�9�@�@�@�@�@�@�@�@������2�5�?�A�5�(�������������G�T�`�f�e�`�W�T�Q�G�;�3�;�?�G�G�G�G�G�G�f�g�n�p�o�l�i�f�^�Z�S�M�B�=�A�B�H�L�Z�f��� �����!�"�!�����������������~�������������������~�r�f�S�R�V�Y�b�r�~��"�/�H�a�w�|�o�g�b�T�;�/�$����
�����������ĽԽн̽Ľ�����������z��������������������������������������������������������ĿĺĹļĿ������ܻ��� �������ܻڻػܻܻܻܻܻܻܻܻ!�-�4�:�?�>�:�-�*�!�����!�!�!�!�!�!�y�����������¿ÿ����������m�\�U�Y�b�m�y���������������ݿܿݿ�����'�*�4�@�I�@�:�4�/�'�!�'�'�'�'�'�'�'�'�'ǔǡǫǡǠǔǈ�{�o�h�o�{ǈǈǔǔǔǔǔǔ�������ʾ׾ؾ־ʾǾ��������������������������ʼּ������ּʼ������������������r�����������������������r�f�\�U�\�f�r�y�������������������y�l�b�`�]�Z�\�`�q�y����)�6�?�C�B�<�6�)�������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÇÓØàìíìëæàÓÇ�|�z�u�w�z�}ÇÇ�a�l�n�n�n�n�a�`�\�]�a�a�a�a�a�a�a�a�a�aŭŹ��������Źŭŧţŭŭŭŭŭŭŭŭŭŭ�������û̻û������������������������������!�$���������������� 4 Y 7 @ C .  L # M } 4 f V , Y 0 F M . =   H C l ( 2 D R U 6 . Y f 2 / a E / : = , 8 = � V 8 ) 9  / 0  r > N E  �  6  J    �  �  �  b  `  :  z  �  �  @  C  D  �  :  �  �  �  M  �    ?  �  �  H  m  P    �  z  �     �  �  (  Q  �  }  �  �  �  3  �  �  h  �  �  I  V  �  �  K  T  e  X  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �    &  6  A  B  =  /    �  �  �  �  a  A  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  u  \  ?  "    �  �  �  s  J     �  �  �  f  6  i  f  d  ]  L  <  (    �  �  �  �  �  u  P  $  �  �  *   �  R  W  [  W  Q  K  E  ?  8  2  ,  %  #  $  #  !          f  ]  U  M  I  F  B  ;  2  )            �   �   �   �   �   �  F  �  '  l  �  �      &      �  �  J  �  �  (  �  �  �  n  �  	  >  r  �  7  �  �  c  �  	  	W  	�  	�  
5  
�  
�    g  �  [  �    /  O  b  e  Q  -  �  �  ~  *  �    n  �  �  �  �  �  �  �  �  �  �  �  �  x  b  K  -    �  �  �  �  �  }  >  9  3  .  )  $          �  �  �  �  �  �  �  �  �  �  u  �  �  �          	     �  �  �  U    �  �  S  �  >  d  l  n  l  l  h  a  W  ]  �  �  �  |  a  1    �  J  �  |  �  �  �  �  �  �  �  �  �  �  w  `  E  )    �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  }  Y  1    �  �  X    �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  �  �  v  X  +  �  �  n  #  �  �  ?    �  �  }    �  �  �  �  �  w  m  b  X  O  G  >  5  ,  "         �  �  �  �  �  �  �  �  |  i  Y  H  ;  /       �  �  �  q  J  !   �   �  Y  X  W  U  Q  L  D  ;  .  !       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  %  �  �  �  T  �  �  �  �  �  �  �  �  w  ^  =    �  �  �  n  G  &  �  �  �  	  	  	&  	-  	,  	  �  �  �  `  %  �  �  -  �  �    �  �  �  $  @  C  6  !  	  �  �  �  �  �  Q    �  '  �  	  ;   �    �  �  �  �  �  �  �  �  k  U  ?  (    �  �  t  "  �  m  �  �  �  �  �  �    }  {  x  t  n  h  c  ]  W  Q  K  F  @  �  �  �  �  �  {  U  &  �  �  d    �  `    �  f    |  ;  #  R  Z  S  C  .    �  �  �  �    _  =    �  �  �  �  x  f  R  ?  *    �  �  �  �  �  b  7    �  �  i  ,  �  �  n  r  q  p  o  j  c  ]  Q  B  3    �  �  �  m  ?     �   �   {  q  j  c  \  U  S  a  n  {  �  �  �  �  �  �  �  �  �  �  �  A  Z  l  q  r  k  T  :    �  �  �  G  �  �  *  m  �  �  �  %        �  �  �  �  �  �  �  �  �  x  g  G  !   �   �   �  �  z  y  y  u  o  h  _  O  7    �  �  H  �  �  C  �  j   �  �  �  �  �  c  D  %    �  �  �  u  @    �  �    �    n  6  :  ;  3       �  �  �  c  )  �  �  #  �  �     �    �  V  �  �  �  �  n  Q  0    �  �  �  D  �  �  C    �  �  !  �  n  ]  N  '  �  �    <  �  �  V    �  W  �  �     �   �  I  F  D  B  ?  =  ;  8  6  4  0  ,  (  #            	  �  �  z  �  �    &  =  F  A    �  �  9  �  p  �    �  J        !  )  '         �  �  �  �  �  �  y  c  Q  D  >  �  �  �  �  �  �  �  �  �  �  �  p  X  <       �  �  �  c  �    :  Q  _  c  b  ]  Q  <    �  �  D  �  �  
  r  ]      �  �  �  �  �  �  �  �  �  ~  q  d  X  K  ,    �  �  �  |  |  {  r  h  W  E  /    �  �  �  �  p  F    �  �  �  i  k  e  _  Y  T  N  H  @  8  /  &        �  �  �  �  �  l  �  �  �  �  �  �  �  f  H  %    �  �  �  �  �  �  _  B  :  l  �  W  �  �  �  �  �  k  "  �  ]  �  4  m  
g  	+  �  �  �  �  �  u  _  B    �  �  �  U    �  �  7  �  �  X  �  W  �  J  Z  j  t  }  �  �  �  �  ~  f  X  G  ,  �  �  H  �  f  ?  5      �  l  �  Q  �  �  �  I  �  -  >    �  �  <  �  I  0  :  <  9  -  $    �  �  �  �  j  /  �  �  J  �  �    �  (  �  �  �  �  ,  F  V  [  T  <    �      �  H  
  �  ,  �  �  �  �  �  x  T  .    �  �  �  J  �  �  J  �  �  )  �  J  8  %       �  �  �  �  �  �  �  �  �  !  b  �  �  #  c  �  �  �  u  h  [  O  F  ?  8  0  )  "    �  �  {  K     �  �  z  ^  [  W  P  H  ?  9  7  E  e  b  I  .    �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  u  f  V  G  7  '