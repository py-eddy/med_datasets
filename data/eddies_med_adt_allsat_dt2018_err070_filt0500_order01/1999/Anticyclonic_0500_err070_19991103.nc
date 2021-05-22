CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ԋC��%      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N$�   max       P��       �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �0 �   max       >O�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F(�\)     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vh�\)     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�`          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >�9X      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0q�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0~�      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >'�|   max       C�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >8�i   max       C�	�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         B      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N$�   max       P��^      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���'RTa   max       ?���)_      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �0 �   max       >%�T      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F(�\)     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vg��Q�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�ܠ          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G   max         G      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�Q��     p  S(                     
   
         (      $      2          I   I   0  A                  E         F               m      0         	            
      i   %   	      �      
         B            	NA��N*��OۉN�6�NI5O���Ng��N��O��NY��Ow��N�k
Oda�N\�LPq�N��4O�k�P6�P�� O��mP���OiE�NG`�Ob�NT�N��P@�N�N���P.3LN]d�O_�ENˏ�NX�O�dEN �4O�ҬNt�[ObN��N��N,+]N�.�N8
O *�PU�O��.N��O̕BOкQO���O�Ov�$OʊO���N߸<NH4KNhܛN$ν0 ż�1�e`B���
��o��o;ě�;ě�;ě�;�`B<49X<49X<T��<�o<�t�<�t�<�9X<�9X<�`B<��<��<��<��=+=+=+=C�=�P=,1=,1=,1=8Q�=8Q�=8Q�=8Q�=<j=@�=@�=@�=@�=@�=H�9=L��=L��=P�`=T��=Y�=]/=aG�=aG�=m�h=m�h=u=���=�{=�9X=��=�l�>O���������������������NB?<BDO[[\[NNNNNNNNN�������
%#�������������������������.0<IMTUUUI@<20......��",/;FMNMHD;/"	�3028<HKNKH?<333333333.69BOWXTOGB=6333333[WW[[^bht�����}th[[�������������������������������������������������������������
#/<ERYSHA</#�')))������BP[t����������th[QHB,./07<IQUUVUOPI<20,,YUV]fmrz��������zmaYI]gt�������������tQI����5F[`iqmpgC�������������������������)B[t�������gNB)����  ����)*,,)�����
#*9=?</#
����������������������((*6>CLOPOJC66,*((((�������)5KR9����edd`cht�������theeeehopjh[OFB=<ABO[hhhhh�������
*)!
����������������������������������������������������


�������������������������HILanz���������z_QIH����������������������������������������~�������������~~~~~~������������������������������������������������������������"(/4;>;2/)")/6:BELJB6))OHJO[]hhha[OOOOOOOOO�������������������������)7AA6�����������
/<EKH</#
��������������������������&)5NjeNB5)�����������
#%(("
������������������������������������������������������

�����rnjqt������������|tr�����

��������������������������������%�����������������������������uvz�����|zuuuuuuuuuu�����������������������������������������l�i�l�y�����������y�x�l�l�l�l�l�l�l�l�l�5�A�Z�g�s�����������s�Z�N�A�5�'�&�%�(�5�������!�-�4�-�+�!������������ﻞ�����������������z���������������������T�a�m�s�x�z�{�w�m�a�T�;�2�/�(�!�+�;�I�TD�D�D�EE	E	ED�D�D�D�D�D�D�D�D�D�D�D�Dߺ������������������������������!�-�:�F�H�S�_�j�h�_�^�S�F�:�2�-����!�/�;�H�T�V�V�T�H�;�/�/�*�/�/�/�/�/�/�/�/�A�M�Z�]�a�e�f�Z�M�A�4�(������(�4�A�/�<�H�R�T�H�E�<�4�/�$�#�� �#�$�/�/�/�/�������������������������������Z�]�f�f�f�c�Z�O�M�M�G�J�M�S�Z�Z�Z�Z�Z�Z�Z�s�z�~�z�q�f�Z�A�(���������4�Z�f�r��������������r�q�f�Y�T�V�Y�Z�f�f�{ŇŔŠŭ����������������ŭŘŇ�x�t�r�{����"�+�/�1�-�)�"�	�������������������������	�������ƳƁ�h�C�6�-�C�O�:�\ƎƳ��àù������������������øìàÐÇÅÇÒà����)�6�?�H�K�G�@�%��������ðïõ�����;�G�T�[�g�c�`�T�G�.�"����������"�.�;�ʾ׾����׾ʾ��¾ʾʾʾʾʾʾʾʾʾ��;�H�T�a�c�f�g�e�a�H�;�/�$�"�!�*�/�2�9�;�����������������������������������������y�����������y�p�m�i�`�]�`�`�m�r�y�y�y�y���������������������������������������r���������������������r�o�o�n�r�r�r�r��ݿѿĿ������Ŀɿѿݿ������������5�B�f�t�~�g�[�5������������5¿��������¿²¨¦¦²³¿¿¿¿¿¿�4�A�M�Z�[�j�r�v�s�m�f�Z�M�J�A�2�+�)�*�4�f�s���������s�g�f�f�Z�X�X�Z�a�f�f�f�f�������ʾ˾׾ؾ׾ʾ����������������������ùܹ���������ܹϹ��������������������a�n�r�n�m�g�a�a�`�Z�Z�Z�a�a�a�a�a�a�a�a�ܻ���'�4�M�Y�M�=�4����ܻ׻л˻ɻлܼ@�C�G�M�P�M�M�@�4�0�3�3�4�8�@�@�@�@�@�@�'�,�'�%�&����������ݻ��������'�m�y�}�����{�y�q�m�d�`�_�`�e�m�m�m�m�m�m�ûл׻ػлĻû��������»ûûûûûûû��#�,�0�9�0�,�#������#�#�#�#�#�#�#�#�Y�f�r�������w�r�g�f�Y�M�F�I�M�M�X�Y�Y�'�4�@�E�B�@�5�4�3�'��&�'�'�'�'�'�'�'�'���������������������������������������������������������~�Y�O�M�F�3�*�(�+�3�L���m�r�y�}�����������y�`�T�G�D�D�G�T�`�e�m���������Ľн�����ݽнĽ��������������������������������y�m�P�K�I�P�`�m�y����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDrDsDyD��N�Z�_�`�b�g�i�Z�N�A�5�(������(�A�N�����
������
�����������������������/�<�H�U�a�zÇÈ�~�y�o�a�H�<�2�!� �#�)�/�������#��������߿ݿѿοݿ���E�E�E�E�E�E�E�E�E�E�EuEnEhEeEiEuE�E�E�E��l�x���������������������x�l�_�S�O�S�[�l����������������������������������������ĦĳĿ������ĿĳįĦĥĦĦĦĦĦĦĦĦĦ�����Ǽ��������������������������������� R S 4 T b ' < : 8 2  ? P p ! $ U 9 A : 7 A E H h D 0 8   0 j ? * k ) � m l ] R Y 3 R H P < V y v   = . F K + ~ s ? E    f  :  �  �  �  G  �  �  O  r  �  �  �  �  B  �  e    S  f  �  �  J  �  q  �  /        �  �  �  �  6  �  �  �  m  �  ?  B    D  A  �  �  �  A  �  C  U  	  t  0  j  �  x  �t��e`B<�C�;ě���o<���<u<u<�/<t�=L��<���=D��<�9X=�C�<�h=]/=��=���=���>�9X=D��=\)=P�`=��=�P=��=,1=u=�S�=Y�=�+=ix�=P�`>�-=ix�=\=Y�=y�#=e`B=Y�=P�`=�\)=u=]/> Ĝ=�E�=�%=��w>Y�=��
=�C�=�-=�v�>��=��=�"�=�F>�B"3�B��B�B �dB&qLA��B�BK#B��B�B��B�B"�B��B^XB&\�B TvB
�ZB�GB�B��B��B��Be�B 2�B0q�B��B?�B~B�(B/�B5dBϜB�BO�B��B>XB��B!�+B!UAB"�;A�jjBq�B"�B�PB�0B��B!��BͨB�B@gB�B��B
r/B�B,�B.�UB#�B��B":�B��B��B �=B&��A���B��B=�B0ABgrB�RB��B��B8jB��B&C�B ��B
�MBM(B��B�PB��B�WB@�B��B0~�B�B�B�&B�ZB�B8�B�&BD�B@�B��B>�BȆB!�[B!@B"�bA�ňB@IB=~BZB@rBB=B!@�B@3B>�B�B �B?WB
A�BdB,¢B.��B$ BI`@ T#A��A��@` �@�,"A�X-C�@�@Y"@���A���A:�6A���A�$7A=�)A:û@�?CA��A��B�A���AӧoAan�ASjA�K�A�)'Al��A���@�A|rA��A�ՄA=*GABy�AN��>'�|A�>�@�
�@Щ�@��sAl-�@��pA�-@��@���A���?��JAj��A&��AoF5C���A��A���A�m?A���C�@��A!2�A�@��=@#W�A��A��m@_,�@��0A���C�E�@[��@��)A��4A;O2A�wA�A>�A:�-@�.�A��4A��CB�A͋�AӃ^Ac�AS��A��A��HAk��A�� @�_-A{%�A���A��A<�NAB��APP�>8�iA�f�@���@��}@��pAk�@�DA�@�
Z@ͶEA�}'?��Ak�A"��Aj�C��,A�s5A��A�A� \C�	�@��7A �\A�i@�6{                      
   
         (      $      3      !   I   I   1  B                  F         F               n      1         
                  j   %   
      �               B            	         #                                    %      #   +   G      5                  -         -               #      !                           -         %                                       #                                          !   '   5      !                  %         !                                                         %                              NA��N*��OۉN�6�NI5OB�Ng��Nc �Ob�NY��O;n�N�gOG�WN\�LO��N�O���P�P��^Ot�/O��%OiE�NG`�Ob�NT�N��P�bN�N���O��uN]d�OQI]Nˏ�NX�O�z�N �4OI�Nt�[N���N��N��N,+]N֒�N8
O *�O��\OzRTN��O̕BO(�O���O	+MOv�$OʊO*ON߸<NH4KNhܛN$�  F  �  �  &  �  e  �  �  9  C  �  3  S  \  �  &  �  �  �  �  �  ;  _  E  �  8    �  �  )  �  �  �    �  A  �  �  O    �    �  .  $  
�  *      �  ~    �  4  ]  "  	    ƽ0 ż�1�e`B���
��o;�`B;ě�;�`B<o;�`B<�t�<u<�o<�o=o<��
<���<�=<j=�w>%�T<��<��=+=+=+=@�=�P=,1=}�=,1=<j=8Q�=8Q�=�7L=<j=e`B=@�=D��=@�=@�=H�9=T��=L��=P�`=���=q��=]/=aG�=��=m�h=q��=u=���=��`=�9X=��=�l�>O���������������������NB?<BDO[[\[NNNNNNNNN�������
%#�������������������������.0<IMTUUUI@<20......	"/;DEFC;3/,"	3028<HKNKH?<33333333506>BORSPOEB?6555555YY\_cht}�����{tphb[Y������������������������������������������������������������
#/<BHPVP></#�')))������NPht����������h`YRQN./01<CIJQQJI><<0....]knqz�����������zmb]XPUagt������������gX�����0=MTZWN5������������������������+)),5BNg{����~g[NB5+����  ����)*,,)�����
#*9=?</#
����������������������((*6>CLOPOJC66,*((((������9@D90����edd`cht�������theeeehopjh[OFB=<ABO[hhhhh�������
����������������������������������������������������


�������������������������RPQUan����������znZR����������������������������������������~�������������~~~~~~������������������������������������������������������������"(/4;>;2/)").68BBHDB61)OHJO[]hhha[OOOOOOOOO���������������������������������������
#/3<AGC</#�������������������������&)5NjeNB5)����������

�������������������������������������������������������

�����rnjqt������������|tr��������

���������������������������%�����������������������������uvz�����|zuuuuuuuuuu�����������������������������������������l�i�l�y�����������y�x�l�l�l�l�l�l�l�l�l�5�A�Z�g�s�����������s�Z�N�A�5�'�&�%�(�5�������!�-�4�-�+�!������������ﻞ�����������������z���������������������T�a�l�m�s�s�m�a�T�H�;�4�/�-�/�5�;�C�H�TD�D�D�EE	E	ED�D�D�D�D�D�D�D�D�D�D�D�Dߺ������������������������������-�:�F�S�_�h�f�_�[�S�F�:�5�-�#�!��!�&�-�/�;�H�T�V�V�T�H�;�/�/�*�/�/�/�/�/�/�/�/�(�4�A�M�Z�^�b�`�Z�M�A�4�(�&�����#�(�/�<�H�I�L�H�B�<�/�$�$�)�/�/�/�/�/�/�/�/������������������������������Z�]�f�f�f�c�Z�O�M�M�G�J�M�S�Z�Z�Z�Z�Z�Z�M�Z�r�r�m�f�Z�M�A�4�(������(�4�A�M�f�r�w�����������r�f�c�Y�Y�Y�_�f�f�f�fŔŭ������������������ŹŭŝŔŇ�|�{ŇŔ���������'�+�,�*�"��	����������������Ƴ���������������Ƨ�h�\�T�U�\�T�Y�fƃƳàìù������������������ùìà×ÏÏÓà������*�2�7�5�)��������������������;�G�T�[�g�c�`�T�G�.�"����������"�.�;�ʾ׾����׾ʾ��¾ʾʾʾʾʾʾʾʾʾ��;�H�T�a�c�f�g�e�a�H�;�/�$�"�!�*�/�2�9�;�����������������������������������������y�����������y�p�m�i�`�]�`�`�m�r�y�y�y�y�����������������������������������������r���������������������r�o�o�n�r�r�r�r��ݿѿĿ������Ŀɿѿݿ������������)�5�B�[�c�i�l�m�e�[�N�5���������
��)¿��������¿²¨¦¦²³¿¿¿¿¿¿�4�A�M�Z�Z�i�q�s�u�s�l�f�Z�M�A�4�,�*�,�4�f�s���������s�g�f�f�Z�X�X�Z�a�f�f�f�f�������ʾ˾׾ؾ׾ʾ������������������������ùϹܹ����ܹϹù������������������a�n�r�n�m�g�a�a�`�Z�Z�Z�a�a�a�a�a�a�a�a�лܻ������'�3�*�������ܻջϻλм@�C�G�M�P�M�M�@�4�0�3�3�4�8�@�@�@�@�@�@����#�%�����������޻��������m�y�}�����{�y�q�m�d�`�_�`�e�m�m�m�m�m�m�ûл׻ػлĻû��������»ûûûûûûû��#�,�0�9�0�,�#������#�#�#�#�#�#�#�#�f�r�������u�r�f�d�Y�M�K�L�M�O�Y�Y�f�f�'�4�@�E�B�@�5�4�3�'��&�'�'�'�'�'�'�'�'�����������������������������������������~�����������������~�r�e�Y�C�@�>�@�L�e�~�`�m�y���������������y�`�T�L�I�H�O�T�`�`���������Ľн�����ݽнĽ��������������������������������y�m�P�K�I�P�`�m�y����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��N�Z�_�`�b�g�i�Z�N�A�5�(������(�A�N�����
�������
���������������������/�<�H�U�a�zÇÈ�~�y�o�a�H�<�2�!� �#�)�/�������#��������߿ݿѿοݿ���EuE�E�E�E�E�E�E�E�E�E�E�E�E�E|EuEsEoEqEu�l�x���������������������x�l�_�S�O�S�[�l����������������������������������������ĦĳĿ������ĿĳįĦĥĦĦĦĦĦĦĦĦĦ�����Ǽ��������������������������������� R S 4 T b " < 3 3 2  ? Q p  / R 0 7 9  A E H h D  8   * j = * k ' � J l V R Y 3 L H P ) I y v  = ' F K  ~ s ? E    f  :  �  �  �  �  �  m  0  r  �  �  �  �  j  �  �  �  �  �  �  �  J  �  q  �  H      �  �  �  �  �  �  �  �  �  	  �  ?  B  �  D  A  a     �  A  f  C  +  	  t  o  j  �  x    G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  F  C  ?  ;  5  /  '        �  �  �  �  �  �  �    >  f  �  �  �  t  [  A  0  $        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  M  1      �  �  �  j    S  &      �  �  �  �  �  �  �  q  i  j    �  �  ]  I  @  9  �  �  �  y  l  _  R  E  8  *         �   �   �   �   �   �   �  	  #  ;  P  ^  e  d  Y  F  /    �  �  a    �  L  �  J   �  �  �  �  �  �  �  �  p  S  3    �  �  �  h  ;    �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  7  9  7  /  "    �  �  �  �  �  ]  2    �  �  �  z  �  C  ?  :  6  2  -  )  $              �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  p  0  �  �  9  �  }  '    @  �  �    "  .  2  2  -  #    �  �  �  �  �  `  6    �  �  G  Q  R  D  *    �  �  {  C  �  �  n  2  �  �    j  �    \  g  r  }  w  o  f  X  G  6  #    �  �  �  �  �  �  u  ^  �  "  G  i  �  �  �  �    d  C    �  �  i  �  `  �  �   z         $         �  �  �  �  �  }  `  A    �  �  �  �  y  �  �  �  �  �  ~  j  L  +  
  �  �  �  K  �  �  �  p  `  |  �  �  �  �  ~  e  H  +      �  �  d    �  n  �  W  �  l  �  �  �  �  �  �  �  d  8    �  �  Y    �  )  |  y  %  |  �  �  �  �  �  r  O  #  �  �  �  `  #  �  b  �  +  $  �  F  �  S  n  <  �  [  �  �  �  1  �  �  �  :  l    �  W  �  ;  :  8  6  7  5  +      �  �  �  �  v  K    �  �  `    _  U  K  B  8  .  #         �  �  �  �  �  �  r  W  =  "  E  A  :  .          �  �  �  �  p  B    �  �  E  �  �  M  �  �  �  �  �  �  �  �  �  �  �  }  k  Y  G  5     �   �   �  8  2  +  $          �  �  �  �  �  �  �  �  �  �  �  �    @  g  w  {  n  X  :  "    �  �  �  6  �  O  �  x  �  �  �  �  �  �  �  �  �  �  }  s  i  `  V  L  B  8  1  *  #    �  �  �  �  �  �  �  �  s  V  1    �  �  {  J    �  �  D  M  �  �    %  (  &    �  �  y  6  �  �    �  �    3  R  �           �  �  �  �  �  z  Y  7    �  �  �  7  s  �  �  �  �  �  �  �  �  t  c  K  .    �  �  �  `    �  )  �  �  �  �  �  �  �  ~  m  Z  D  %  �  �  �  R    �  �  t  ;              "  %  (  *  ,  .  /  1  3  5  7  :  <  >    c  �  �  �  �  �  g  5  �  �  !  �  
�  
1  	b  V  �    �  A  )    �  �  �  �  d  8    �  �  z  F    �  �  e  �  {  R  O  P  �  �  �  �  �  y  G    �  z    �  �  �  �  j   �  �  �  �  �  �  �  �  u  e  V  A  '    �  �  �  �  3  �  s  '  E  F  7  '    �  �  �  �  �  m  ?  $      �  �  �  �    �  �  �  �  �  �  �  �  �  �  w  g  T  9    �  �  �  �  �  �  �  �  �  y  j  Z  K  <  )    �  �  �  �  �  �  e  K          �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  |  �  �  }  u  c  H  *    �  �  �  b  -  �  �  5  �  '  �  .  +  '    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  $          �  �  �  �  �  �  �  �  �  �  �  �  y  h  V  	�  	�  	�  	�  	�  
  
Z  
x  
�  
|  
J  
  	�  	J  �  �  �  �    "  �  �    *    �  �  �  �  J  �  �  (  �  '  �    ~  �  �      �  �  �  �  �  �  �  �  �  n  P  0    �  �  v  >        �  �  �  �  ~  U  '  �      �  �  �  n    �  u  �  �  %  �  �  5  w  �  �  �  �  �  �  *  �  �  n  �  8  u  �  ~  w  j  S  6    �  �  �  u  G    �  �  b  !  �  �  Y  S    
          �  �  �  �  �  �  v  X  6    �  �  D  �  �  }  p  X  :    �  �  _    �  �  �  �  �  :  �  �     7  4         �  �  �  �    H    �  �  r  K  *  6    �  �    �    6  T  \  R  ?    �  �  C  
�  
$  	x  �  �    ,    "  	  �  �  �  �  �  n  P  0    �  �  �  X  %  �  �  }  H  	  �  �  �  �  �  �  �  y  _  J  9  (    �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  p  _  M  7  "      &  1  �  �  �  �  q  Z  A  '    �  �  �  �  G  �  �  a    �  h