CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��-   max       Pa�?      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @E������     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vw33334     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�9�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >x��      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Zk   max       B,      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��"   max       B,#      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�ܶ   max       C��#      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�u`   max       C���      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          3      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          )      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��-   max       Pr�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�8�YJ��   max       ?������      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       >�P      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @E������     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vw33334     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >4   max         >4      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1��   max       ?��6��     �  M�            
      !      $                     '      
         .   -   H   4      F         
         [      6      I                  
      	      !   �         	            N��AN��vN|��N�g#N^-O��VN��O��N5tN"�Nv8!O���O�y_O��O�?)N{AnOE?mO�{N�7�P-�O֞O�2�P&(�O��O� �N	��N�<Njo�O��O#�APW�O�"Pa�?N�p�P��OwނM��-N��[N��NR>�N�"�OP Nj��O_K�O��nO��tO3NC�N\%�Nݷ�N��=N�߅N'ü��j��j�T���ě��ě�;D��;�o;��
;��
;ě�;�`B;�`B;�`B<D��<D��<u<�o<�o<�C�<�C�<��
<�1<�9X<�j<���<�/<�/<�/<��<��=o=t�=�P=�P=��=8Q�=D��=D��=L��=aG�=e`B=m�h=�7L=�7L=�C�=�\)=���=���=�9X=�E�=�^5=���=�����������������������������������������TRUbijjifeb`_ZVUTTTT��������������������ZUYantwqnaZZZZZZZZZZ�����������������������������������������������
 "
�����������������������^Wannsrnfa^^^^^^^^^^��������������������99BNS[t|�����tg^[NB9^]]__bmt����������t^����
#(/<HQK</�� /<HUenri^UH<4/$�����������45BNQ[cgjljd\NB;8684YQOUV^hr|{�������thYNNW[giqttuthge[SONNN���#U[YNH;0
�����#/<Hant}~znaH/#��������������������������!.>BGFA)��)5B[gpg[VNG5)BEN[gtw}���tg[NKEBB_^agnysqna__________|���������||||||||||�������������������������#(/38<GG/#���JQUafnpz�������znaUJ��� ��)BUUG5)������
 &*-10/+#
����)5WmlbQB.��yv{�������������yyyy�������&)/.'�������������

�����##/0//'###########		
#+-)'#
				HOOH</-//9;<HJHHHHHH��������������������425<HTUanvnjaUH<4444wz{��������������|zwZYanz�~zpnlaZZZZZZZZ���������������������z������������������������

���������������

�������")6876/)fhjt������thffffffff���������]Z[acmqz|}zzma]]]]]]almsz{��zmca`aaaaaaa�


�����������������������������������������'�3�:�@�@�A�@�:�3�'�&������'�'�'�'�x���������x�l�_�S�F�D�F�S�_�l�p�x�x�x�xĦĳĽĿ��ĿĽĶĳĦĚĕĔĔĚĦĦĦĦĦ�����������߾߾�����������N�Z�g�s���������s�j�g�N�A�(�$�#�'�.�A�N��������������������������������������������)�5�>�N�T�P�B�5�)������������������üʼּ޼ּʼ�����������������������������������������������������������������%�*�6�9�@�6�*�������������uƁƅƉƁ�z�p�h�\�O�C�6�*�$�)�6�<�Y�h�u��"�.�;�G�T�`�e�f�`�T�;������������"�;�T�a�h�f�a�Y�H�<�%��	���������	��"���������������������������������ƎƚƝƥƤƚƐƎƁ�u�s�uƁƈƎƎƎƎƎƎ�$�-�-�,�0�7�0�+�$�����������������$�����4�Z�f�y�s�f�Z�F�4�(�����������������������������������������������������������ȼ������r�f�Y�Q�G�?�@�M�Y�f������������������������z�w�}�~�{���������������ùܹ��������ܹϹ����������������׿�;�G�T�X�R�.�"�	����׾ʾ����������`�m�y������z�y�}�y�k�\�G�?�;�9�<�G�T�`�[�h�w�|�y�l�h�[�O�6�)��#�&�*�1�6�B�O�[E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E��n�x�y�t�zÄ�z�n�d�b�a�a�a�c�n�n�n�n�n�n���5�A�F�G�E�A�:�5�(�������������������$�(�2�6�6�5�(�������������y²��������¿�t�g�N�9�/�*�8�9�N�y�m�y�������y�m�`�T�G�;�3�.�'�.�;�G�T�[�m�s�����������������������s�_�A�.�1�9�Z�s���������������������������|������������Ħĳ��������� ��
��������ĿīģĜěĦ�(�4�A�M�Z�f�s��������s�f�M�F�7�*�(��(�f�s�������s�m�f�f�f�f�f�f�f�f�f�f�f�f�������������ź�������������������������¦¦²³µ²¦¦¦¦¦¦�A�B�M�T�Z�\�Z�Q�M�J�A�=�7�8�A�A�A�A�A�A�(�4�A�I�E�A�@�:�7�4�-�(�%�"�#�$�(�(�(�(����!�!��������޻ܻٻ޻����������������������~�~�~�������������������y�������������������y�l�`�S�K�T�`�c�l�y�������������������ּʼ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�DvDvD~D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�ExEuErEpEsEuE�E�E������������������x�o�x�}�����������������'�.�4�:�:�4�,�'����%�'�'�'�'�'�'�'�'�:�F�S�_�_�l�x�x�p�l�_�S�F�>�:�-�%�-�.�:ŭŹ��������������ŹŭŬŢŨŭŭŭŭŭŭ������������ùìãåìôù��������������ǮǭǭǡǔǎǏǔǡǮǮǮǮǮǮǮǮǮǮǮ > J � @   A * V b D d > P 0 R 2 X ' 6 G : O D 6 I M z : n N  0 ' > Y @ , L Z p J D   g - J B T [ " 4 /    �  �  �  �  h  �  .  b  S    �  p  �  �  )  �  �  �  �  �  "    9  l    /  G  �    �  �  �  �  �  �    &  �  �  �  3  o  �  �  �  �  �  [  �    �  �  :�u�#�
�t�;�o;�o=t�<49X=#�
<49X<o<#�
<�j<�j=o=P�`<�t�<���='�<��=}�=�o=�j=��=<j=\<��=o=t�=q��=<j=��#=D��=�Q�=0 �=�S�=�hs=T��=u=�+=�%=��=��-=���=ě�=���>x��=�`B=�Q�=ě�=��=��`=�G�=��#B��B!D�B'�yB�Bp�Bb�B�BH�B ��B�[BٖB	(xB�2BmB�qBN�B\�B pB	�B%0�B5�B��BzBZ B	<B��B�B��BMB�BJ�B@�BU�B
�XB�fB#��BدB$�uB��B��B��B@�B�kB,Bk�B��B�1BD�Bx�BחA�ZkA��*BvaB��B!?�B'�pB8B��BBB�2B?�B ��B�B��B	J�B�B~�B��B;�B�9B@�B	:=B%G1B=B��B|iB5�B	.�B�B-�B��BE�B��BDnB=�B��B
�0B��B#��B�6B$��B�XB>#B?BL�B��B,#B;�B��BˋBA3B��B�A��"A���BP)?P'\?�^�@�mA�ǪAW7<A��AЩ�A��
@���@�))A�;"B�]Aa0:A�wJA�Y�B�HB�A5��A���@���A��>�ܶAT��Ah�bAِlC���C��#A��A���A�#�A��AhA�
�A�{�A�,A>��AC�>@��A��8A;�A8h@�D@s�A�G@��*C��.C� @��V@�l@��A��BAͥ2B��?V+�?�L�@�e�A��AWDA�c�AП�A� $@��@�WfA��8B�\Ab��A���A�B� B	&�A31�A�z@��A��@>�u`AS�kAgEKA��C���C���A�w�A��%A�{#A��Aj�A�4�A�_�A��A@�eAD�@�A� �A<��A8��@�
�@�A)�A>C���C�@�%�@ͫ	@���A��.A̓qB��   	               "      $                     (      
         .   -   H   4      F         
         [      6      J               	   
      
      !   �         	                                                      !            #      )   !   !   -                        3      3      '                              %                        
                                                                                                   )                                    %                        
N{�N4(�N|��N�g#N^-O)�zN��O��N5tN"�Nv8!OuXOu�)Op}QN��hN{AnO�>OA N�7�O�5�O�<O���O89tOe	0O��N	��N�<Njo�O���O#�AO��O�"Pr�N�p�O��OwނM��-N��[N��NR>�N�"�OP Nj��O_K�O��nOmOB�NC�N\%�N��N��=N�߅N'ü  �  �  �  `  �  �  +  O  I  �  �    �  �  T  �  v  b  �  |  �  	  �  I    y  �  S  �  z  	M  &    �  	;  d  6    �  �  �  �  �  ;  Y        :  �  �  }  ]��9X��t��T���ě��ě�<�o;�o<t�;��
;ě�;�`B<o<49X<�C�<�<u<�C�<ě�<�C�<�h<�h=+=L��<���=H�9<�/<�/<�/=+<��=��=t�=H�9=�P=�o=8Q�=D��=D��=L��=aG�=e`B=m�h=�7L=�7L=�C�>�P=�{=���=�9X=�Q�=�^5=���=�����������������������������������������TRUbijjifeb`_ZVUTTTT��������������������ZUYantwqnaZZZZZZZZZZ������������������������������������������������

������������������������^Wannsrnfa^^^^^^^^^^��������������������<BNR[t{�����tg_[TNE<ebeegt������������pe���
#$37;:/#
��+*,/<HMUWUSH?<8/++++�����������=:8:BNT[`gikigc[XNB=[VY[_`htvy}����}thd[NNW[giqttuthge[SONNN�����
#0EOMKF>0#
 �#/<HanqrrnfUH</#�����������������������).21-) �)5BN[a[XUQNB5)LIJNN[gmtuyyutg[YNLL_^agnysqna__________|���������||||||||||��������������������������
#/6:<@/#���JQUafnpz�������znaUJ���)5;@@;3)��
 &*-10/+#
�����)DOTVB5)�yv{�������������yyyy��������������������

�����##/0//'###########		
#+-)'#
				HOOH</-//9;<HJHHHHHH��������������������425<HTUanvnjaUH<4444wz{��������������|zwZYanz�~zpnlaZZZZZZZZ���������������������z�������������������������	

������������


��������")6876/)fhjt������thffffffff����������]Z[acmqz|}zzma]]]]]]almsz{��zmca`aaaaaaa�


����������������������������������������������'�3�5�:�;�3�'�#��"�'�'�'�'�'�'�'�'�'�'�x���������x�l�_�S�F�D�F�S�_�l�p�x�x�x�xĦĳĽĿ��ĿĽĶĳĦĚĕĔĔĚĦĦĦĦĦ�����������߾߾�����������A�N�Z�g�s�s�z�u�s�g�Z�N�A�A�5�2�4�5�=�A��������������������������������������������)�5�:�J�Q�K�B�5�)������������������üʼּ޼ּʼ�����������������������������������������������������������������%�*�6�9�@�6�*�������������uƁƇƁ�y�o�h�\�O�C�6�,�&�*�6�>�O�[�h�u�"�.�;�G�T�a�b�`�Y�T�G�;�.�"��	�����"�/�;�N�T�^�[�J�H�/�"��	����	���"�/�����������������������������������ƎƚƝƥƤƚƐƎƁ�u�s�uƁƈƎƎƎƎƎƎ�������#�'�0�5�0�)�$�����������������(�4�A�M�W�S�M�A�4�(���������������������������������������������������f�r���������������������r�f�X�S�O�Y�f�������������������������������������������ùϹܹ���	�	�����ܹù����������������ʾ׾�����	�������׾ʾž����������ʿT�`�m�{���{�w�v�r�m�f�`�T�G�E�=�;�?�G�T�B�O�[�h�h�p�n�h�^�[�O�B�6�4�1�5�6�>�B�BE�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E��n�x�y�t�zÄ�z�n�d�b�a�a�a�c�n�n�n�n�n�n���(�5�>�C�E�E�C�7�(������������	�������$�(�2�6�6�5�(�������������N�[�g�t�t�g�[�N�D�@�A�K�N�m�y�������y�m�`�T�G�;�3�.�'�.�;�G�T�[�m�Z�s�����������������������q�\�K�?�?�E�Z���������������������������|������������ĳĿ����������������������ĿĹĲİĲĳ�(�4�A�M�Z�f�s��������s�f�M�F�7�*�(��(�f�s�������s�m�f�f�f�f�f�f�f�f�f�f�f�f�������������ź�������������������������¦¦²³µ²¦¦¦¦¦¦�A�B�M�T�Z�\�Z�Q�M�J�A�=�7�8�A�A�A�A�A�A�(�4�A�I�E�A�@�:�7�4�-�(�%�"�#�$�(�(�(�(����!�!��������޻ܻٻ޻����������������������~�~�~�������������������y�������������������y�l�`�S�K�T�`�c�l�y�������������������ּʼ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�EyEuErErEtEuE�E�E�E������������������x�o�x�}�����������������'�.�4�:�:�4�,�'����%�'�'�'�'�'�'�'�'�F�S�\�_�l�v�n�l�_�S�F�@�:�-�)�-�:�;�F�FŭŹ��������������ŹŭŬŢŨŭŭŭŭŭŭ������������ùìãåìôù��������������ǮǭǭǡǔǎǏǔǡǮǮǮǮǮǮǮǮǮǮǮ 4 P � @   A % V b D a B G $ R 5 Y ' , < 2 # / " I M z / n   ) ' ' Y @ , L Z p J D   g ( 9 B T Z " 4 /    �  o  �  �  h  j  .  "  S    �  -  �  �  �  �  `  �  �  �  c  c  �  �  2  /  G  �  M  �    �  �  �  )    &  �  �  �  3  o  �  �  �  :  b  [  �  �  �  �  :  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  >4  �  �  �  �  �  �  �  �  �  �  �  u  e  Q  =  5  0  &      }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  Y  C  )  �  �  �  �  �  �  �  �  �  �  �  �  �      	          `  ]  Z  U  P  I  @  3  !    �  �  �  v  P  *    �      �  �  �  �  �  �  �  �  w  b  L  4    �  �  �  �  �  o  R  Z  �  �  �  �  �  �  �  �  �  �  �  �  z  F  �  �  >  �  d  +  &  !          �  �  �  �  �  �  �  �  �  g  D    �  C  L  O  I  ;  )    �  �  �  M    �  p    �  >  �  b  �  I  K  L  N  N  M  L  I  E  A  <  8  4  0  /  -  +  )  &  $  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  [  S  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  l  I    �  �  ?  �  g   �  �  �  �  �  �  �  �  �  �  �  �  z  X  2    �  �  q  5    |    �  �  �  �  �  �  ~  r  a  L  0    �  �  j  Z  7    �  ?  �  �  �    ,  B  Q  S  D  '    �  w    �  
  x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  u  v  n  d  Z  P  G  ?  7  .  $      �  �  �  v  1   �  Q  K  C  9  T  `  b  `  Z  E    �  �  �  �  x  F    �  �  �  �  �  �  �  �  �  �  y  f  R  @  0  (  )  1  L    �  �  	  D  c  q  y  |  u  c  H     �  �  �  Y    �  a  �  �  N  o  �  �  �  �  �  �  �  �  ^  )  �  �  e    �  2  �  �  �  �  �  �  	  	   �  �  �  �  ]  !  �  �  x  &  �    �  Z    �  �  �  �  �  �    C  q  �  �  �  y  O    �  ?  �            *  G  D  B  >  9  1  )  '    �  �  �  D  �  �  ^    
�  i  �  �  �  �      �  �  �  |  	  
�  	�  	T  �  d  %    y  w  u  r  p  n  k  h  f  c  `  ^  ]  [  Y  R  K  D  <  5  �  �  �  �  �  �  �  �  �  �  �  �  r  Z  A  )    �  �  �  S  A  .    	  �  �    L  ]  A  $    �  �  �  �  i  G  %  �  �  �  �  �  �  v  k  _  L  9  #    �  �  �  I  �  O  6  z  a  J  7  (      �  �  �  W  '  �  �  �  �  �  a  E  \  �  A  �  J  �  �  		  	*  	C  	L  	>  	"  �  �  4  �  �  �  �  K  &      �  �  �  �  �  k  I  &    �  �  �  t  F  6    �  �  �  �  
      
    �  �  �    =  �  �  M  �  W  �  �  �  �  {  t  f  U  E  4  "    �  �  �  �  �  �  �  �  �  }  ?  �  �  	  	2  	6  	0  	;  	  �  �  u  2  �  [  �  )  l    x  d  \  R  I  6  "    �  �  �  �  �  M    �  ~     �  >  �  6  (      �  �  �  �  �  �  �  �  y  c  M     �  �  �  P      �  �  �  �  �  �  �  �  h  I  "  �  �  �  ~  U  $  �  �  s  %    �  �  �  �  �  �  �  �  �  {  l  \  I  4      �  �  |  b  F  *    �  �  �  q  D    �  �  �  {  D  
  �  �  �  �  �  �  �  t  \  C  )    �  �  �  �  ^  E  +    �  �  �  y  ]  =    �  �  �  �  Z  -    �  �  g  2  �  Y  �  �  �  �  }  t  e  S  @  ,      �  �  �  �  h  3  �  �  �  ;  7  4  .  '        �  �  �  �  L    �  �  �  �  x  G  Y  1    �  �  r  9  �  �  �  �  �  {  :  �  �  g  �  c  �  �  )  �  �  +  u  �  �      �  �     4    �  F  �  �  
�  �  �  
  �  �  �  ~  J    �  �  J  �  �    {  �    �   �    
  �  �  �  �  �  �  �  {  ^  5  �  S  �  �  �  C     �  :  ,          �  �  �  �  �  �  �  �    #  @  `  �  �  �  �  �  �  �  �  �  �  d  7  	  �  �    S  &  �  �  �  �  �  �  �  q  X  <  '    �  �  �    ^  ;    �  �  �  l  .  }  j  X  E  :  1  '      	  �  �  �  �  �  �  �  \  /    ]  A  %  
  �  �  �  �  �  v  ^  F  /    �  �  �  �  b  <