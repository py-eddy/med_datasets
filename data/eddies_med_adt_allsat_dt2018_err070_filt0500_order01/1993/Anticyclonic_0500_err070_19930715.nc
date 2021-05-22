CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?Õ�$�/      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��w   max       P���      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       >         �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E�          �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vqG�z�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��`          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >gl�      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+�l      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B+��      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C�en      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��%   max       C�g�      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��w   max       Pe�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��j~��#   max       ?��t�j      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       >o      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�
=p��   max       @E�          �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vp��
=p     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�`          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?�����     �  K�                     
                  1               +   (                        ]   @            G   &      T                                    "   �         	N-g N�FN=�1N���Oj�OLVN�m�O(�N��+N���N��N��wPNv�N]u�N$�O���O��O��3Og��NE8�O���O�Q�N�=O��O�J<O���P��P���Np�O��@O�^VP�O�r�N�Y�P�v�O-��O�7�Nj��NekcN7��M��wN�vN�NjLnO��O�O,0�O���OB0N~#|NGB��9X��9X��`B���
��o�D���D����o%   ;o;��
;ě�;�`B<o<t�<49X<D��<e`B<e`B<e`B<u<u<u<�o<�o<�C�<�t�<���<���<���<���<���<�/=C�=C�='�=0 �=0 �=8Q�=D��=q��=u=�%=�%=�%=�+=�\)=��=�{=�{>   c_gtv����tngcccccccc����������������������������������������#$,01<?G><0#����������������������������������������)6BEKDB6);;AABO[chpt|vtjh[OB;  #/<@HHHHH</+##    zz��������������}zzzX[`gt�������utrge\[X������

����������5BNvseaN5�������	

���������������������������SQW^h���������tjthfS!#&/<AHNTTLH</,%##!!�����)/5A5#�����<;:=BBN[gjprrpog[NB<�����������������������).5<BIPK5)	�032:>N[gilmmkg[NJB50�������


����������������������������������
������*/<HUaz������znaH/*������������������������)9FMK6����������������������������	�/06B?;8)������5>C@B=5)���ooq|��������������zo����������	�����TUXaamwz{zzomaTTTTTT����5B?BW[WB)���������������������������������������������v{�������������{vvvv()./<?DHTH</((((((((����������������������������������������H<:72<@IKOIBHHHHHHHH
)46960*)

9<=ABGOW[\[[ZOIB9999�������������������������������������������������������������������
!
�����
)5BKNFB<53)OS[gltztg][OOOOOOOO��������������������ÇÓÔÕÓÈÇ�z�x�u�zÄÇÇÇÇÇÇÇÇĚĦĳĸĳĪĦĚĘĖĚĚĚĚĚĚĚĚĚĚŇŔřŘŔŇ�{�v�{�}ŇŇŇŇŇŇŇŇŇŇ�������������������������y������������������������)�-�)������������������������*�6�<�C�I�C�B�6�*������� ���zÇÓÚØÝ×ÓÇ�z�z�u�w�w�z�z�z�z�z�z�
��#�/�2�F�E�<�9�/�)�#���
�����
����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������ûϻлܻ����ܻлû����������������������ƿ������������y�`�;�,�+�9�G�m�������)�1�)�(������������������T�a�c�e�a�T�H�D�H�K�T�T�T�T�T�T�T�T�T�T�M�Z�s�������������f�M�4�(�-�9�4�'�'�4�M�������������������������������/�;�T�a�p�x�x�s�m�a�T�L�/�"������	�"�/�B�O�[�h�t�tĀĀ�w�t�h�[�O�B�6�.�-�0�5�B��"�.�/�2�.�"�������������ƎƚƧƫ������ƭƧƤƚƎƁ�w�o�p�uƁƌƎ�����������������������ƶƼ�������̾������ʾ׾���׾;ʾ����������������������������Ŀ���������������������������������(�A�f�x�y�p�Z�A�(��������������������������������������������������ܹ�����'�)�%�����蹶�����������¹��/�a���a�]�]�d�T�/�	�����������������	�/ÇÓÚÛÕÓËÇÁÅÇÇÇÇÇÇÇÇÇÇ���׾���;�I�I�R�G�;�3�6�"��׾ž��������
�#�0�I�R�`�f�h�b�U�<�"������
���
�x�������ܻ�������ܻлû������q�m�n�xčĚĦĳ��������������ĳĚčą�x�t�~āč����������������������������������������¿�������	��
�����t�^�P�K�[�t±¿��(�4�A�M�Q�Z�`�f�l�f�Z�M�A�4������(�4�A�M�Z�f�p�t�i�V�A�(���	�����(�����!�)�-�5�-�!���	�� �������(�4�A�C�G�A�4�0�(�#�!�"�(�(�(�(�(�(�(�(�(�.�5�A�N�O�Z�N�A�;�5�(�&� �(�(�(�(�(�(�
����!���
��������
�
�
�
�
�
�
�
�ɺƺɺֺ�������ۺֺɺɺɺɺɺɺɺɺr�~�����������������������~�}�r�q�j�r�r�������ɺͺɺƺ��������������������������zÇÓàìð÷ùúùðìàÓÇÆ�|�z�y�z�l�y�����������������y�r�l�`�`�\�`�f�a�l�����ʼּ�������ּʼɼ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DD�D�D��s���������������������������v�s�j�j�n�s����������������������������������������EuE�E�E�E�E�E�E�E�E�E�E{EuEsEuEuEuEuEuEu T ; 1 5 H D K 1 ) 9 A @ d p - : % B " 4 4 - m , s F 0 I ? _ M 8  ; B ` 2 � O � ` R E C ! 9 9 + F ] |    N  ,  N  �    L  �  l  �  �  &  �  �  �  ?  �  ;  R  �  S  .    �  ]  �  �  d  ;  0  �  �  }  �  �  5  �  w  �  �  �  3  H    �  7  <  u  �  �  �  ���C�����o%   <e`B;D��;�`B<D��;��
<�o<�o<D��=e`B<49X<e`B<�h=�P=ix�=Y�<�t�<�1<�/<�/<�/=t�=8Q�=�S�=��<�9X=<j=,1=ě�=��=8Q�=��=y�#=�\)=P�`=H�9=Y�=�%=�o=���=�O�=��T=�-=���>gl�=�
==�v�>	7LB	�[B�BaB%��B�BȈB�B(3By>BV�B	�XB#��B`)B�hB�]BeaB�OB`lB�hB��B]gB�9B1$B�CB"�iB`B.�B�B��B�"B�,B�JB �A���BeB��B�8B)aHB�pB)�BDSB&y�B�.B��B!j�B+�lBBB��B	!B=TB	�.B��B�8B&:�B+�B�qB��B?�B�AB@;B	��B#��B>�B�:B�B�oB��B@B��B~�B;�B�B@BB#��B��B?�B:)B��B<vBB>B�=B@�A�}�B=�BFWB��B)�gB��B�B;B&b[B�5B��B!��B+��B@"B?�Bz�B	B�B��A�tA�qrA�@���AЗ8A�ӝA�ƨA�=�A�?�C�enA��5@��+Al��A�A���AB$�A�Z�A��A�"�A^��B�BjAO>`Ate�A8�A���?�A�A�A��AX�QA��@��7A���A�ܑA��A9�bA9Ѹ@h�lA8A��A��\@?�V@@%1}A��[AQ@��C��A��A�<�C��<A�V�A��XA��@�A�x'A���Aɏ�A��A���C�g�A��_@�
�AiA��A��;AC�A�m�A��A��A_�Bv�B�NAP�DAsfA=EA��>��%A�uAɩXAV��A쁇@��A�}�A�J�A��'A;8A8tl@d9�A6�UA�p�A���@C��@�+@$�A�A��A �GC��4A�.�A�|�C�                                       2               ,   (                        ^   @            G   '      U         	                           #   �      	   
               #                        5         '      %                     -   !   %   I      )   %   %         9                                                                                       1                                    )         /         %            #                                                N-g N�FN=�1N���NPU�OLVN�L2N��N��+N�EN�;N�#P>�"N]u�N$�Oz��N���O��O'UNE8�O)�yO1�NZ�AO��Oʱ�OB�FO~�Pe�Np�O��O�^VO>�7O�@eN���P�O-��O�7�Nj��NekcN7��M��wN�vN�NjLnO��O�O%,O(&O4�dN~#|NGB  �  �  i  q  0  "  �  y  �  �  |  �  �  +  ~  �  �    �  �  �  �  |    r  Y  
=    �  �    �  "    N  \  �    �  �  R  o  �  �    ~  �    +  G  ��9X��9X��`B���
;��
�D���o;o%   ;��
;ě�;�`B<t�<o<t�<u<��
<���<�9X<e`B<�C�<�t�<�o<�o<�C�<���=T��=8Q�<���<�h<���=m�h<��=\)=�%='�=0 �=0 �=8Q�=D��=q��=u=�%=�%=�%=�+=�t�>o=� �=�{>   c_gtv����tngcccccccc����������������������������������������#$,01<?G><0#����������������������������������������)6BDIBB6)$GCHHNOS[hktphb[OGGGG  #/<@HHHHH</+##    ��������������������X[cgnt�������tgg][XX������

������������5ANbm`_N5�������	

���������������������������[\[bhkt��������xth][,*+/<HIPNH@<7/,,,,,,�����(*)'"����@?>ABIN[cglonlg[UNB@��������������������	))145=95+)?;8?BN[egjjjjhg[ZNB?������	


�������������������������������������
������?BKUVanvz~���{znaUH?�������������������������)3;:6)���������������������������

)6:;::95)�����5>C@B=5)��������������������������������������XWZadmuzzzynmaXXXXXX��� )-5GIC5)����������������������������������������v{�������������{vvvv()./<?DHTH</((((((((����������������������������������������H<:72<@IKOIBHHHHHHHH
)46960*)

9<=ABGOW[\[[ZOIB9999�������������������������������������������������������������������

�����)5BHLDB;52)OS[gltztg][OOOOOOOO��������������������ÇÓÔÕÓÈÇ�z�x�u�zÄÇÇÇÇÇÇÇÇĚĦĳĸĳĪĦĚĘĖĚĚĚĚĚĚĚĚĚĚŇŔřŘŔŇ�{�v�{�}ŇŇŇŇŇŇŇŇŇŇ�������������������������y����������������������������������������������������������*�6�<�C�I�C�B�6�*������� ��ÇÓØÖÜÕÓÇ�z�z�w�y�z�ÇÇÇÇÇÇ�
��#�/�<�<�<�;�3�/�#���
���
�
�
�
����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������ûʻлܻ��ܻлû������������������������ÿ¿������������y�T�;�-�,�:�J�m�������)�1�)�(������������������T�a�c�e�a�T�H�D�H�K�T�T�T�T�T�T�T�T�T�T�Z�f�s���������������s�f�Z�A�9�A�B�B�M�Z������
������������������������������/�;�H�T�d�l�m�j�a�T�Q�H�;�/�"����!�/�B�O�[�h�j�t�z�z�t�q�h�[�O�B�8�5�6�7�?�B��"�.�/�2�.�"�������������ƁƎƚƧƳ����ƴƳƧƚƗƎƁƀ�u�s�s�uƁ�������������������������ƿ�������پ������ʾ׾�׾ʾǾ��������������������������������Ŀ���������������������������������(�A�f�v�w�n�Z�>� ��������������������������������������������������Ϲܹ������������ܹϹù������ù��/�?�F�M�O�H�=�/����������������������/ÇÓÚÛÕÓËÇÁÅÇÇÇÇÇÇÇÇÇÇ�ʾ׾����"�.�6�9�)�"��	���׾Ⱦ������
�#�0�I�R�`�f�h�b�U�<�"������
���
�����ûлܻ���ܻջлû���������������ĚĦĳĻĿ����������ĳĦĚčć�{�wāďĚ����������������������������������������¿���������������t�f�_�^�b�g�t��(�4�A�M�Q�Z�`�f�l�f�Z�M�A�4������(�4�A�M�Z�f�p�t�i�V�A�(���	�����(�����!�)�-�5�-�!���	�� �������(�4�A�C�G�A�4�0�(�#�!�"�(�(�(�(�(�(�(�(�(�.�5�A�N�O�Z�N�A�;�5�(�&� �(�(�(�(�(�(�
����!���
��������
�
�
�
�
�
�
�
�ɺƺɺֺ�������ۺֺɺɺɺɺɺɺɺɺr�~�����������������������~�}�r�q�j�r�r�������ɺͺɺƺ��������������������������zÇÓàìð÷ùúùðìàÓÇÆ�|�z�y�z�l�y�����������������y�r�l�`�`�\�`�f�a�l���ʼҼּ��������ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��s���������������������������x�s�k�k�o�s����������������������������������������EuE�E�E�E�E�E�E�E�E�E�E{EuEsEuEuEuEuEuEu T ; 1 5 2 D N + ) 6 G A e p - /  7  4 2 3 l , p . ) 4 ? [ M #  7 F ` 2 � O � ` R E C ! 9 .  D ] |    N  ,  N  �  h  L  �  �  �  �  
  �  �  �  ?  �  �  $  _  S  s  �  �  ]  _  �  �  �  0  W  �  �  ^  �  �  �  w  �  �  �  3  H    �  7  <  R  a  �  �  �  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  �  �  �  �  �  �  �  �  �  �  �  �    r  f  Y  I  8  '    �  �  �  �  �  �  �  ~  |  z  x  v  s  q  m  j  f  c  _  \  i  d  `  [  V  P  K  E  >  6  .  &               (  1  q  l  f  a  [  R  I  @  2      �  �  �  �  �  �  �  �  }  #  .  Y  c  ]  �  �    %  .  /  *         �  �  �  F  �  "             
    �  �  �  �  �  �  �  �  �  �  {  i  �  �  �  �  �  �  �  �  �  �  �  �  |  b  E  &  �  �  t  ,  Z  e  n  u  x  w  r  l  a  R  A  +    �  �  �  �  Z  '   �  �  �  �  �  |  z  x  w  v  v  w  w  u  p  l  g  b  ]  W  R  a  |  �  �  �  �  }  z  e  X  Q  M  <  $    �  �  t  .  �  Z  m  z  v  p  h  `  X  M  B  8  +      �  �  �  4  �  �  �  �  �  �  �  �  �  �  �  �  x  c  N  7       �  �  �  �  �  �  �  �  i  <    �  �  �    �  �  �  N    �    �  �  +  $        	    �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �    s  f  Z  N  A  5  (  )  .  2  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  %  �  �  8  ?  �  �  �  �  �  �  �  �  �  �  |  Z  4    �  �  a    �  p  T  �  �  �  �      �  �  �  �  �  _    �  �    l  |  |  M  j  �  �  �  �  �  �  o  R  2    �  w    �    }  �  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  s  j  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  Z  O  D  8  �  �  �  �  �  �  �  �  �  �  �  �  �  [  1    �  �  v  F  i  u  r  [  ?  !  �  �  �  �  �  s  U  2    �  �  ~  M      
    �  �  �  �  �  �  �  m  T  ;  "    �  �  �  }  H  k  r  o  `  K  :    �  �  �  �  �  �  e  R  .  �  �  �  |  (  0  5  <  E  Q  Y  U  C  *    �  �  �  �  D  �  �  �  �  �  	W  	�  	�  	�  
  
3  
=  
4  
  	�  	�  	)  �  9  �  �  �  =  �  �  �  �  �  �            �  �  L  �    d  �  �  6  (  �  �  �  �  �  �  �  �  �  �  t  `  K  7  %       �  �  �  Q  M  e  �  �  �  |  i  ^  X  Q  =    �  �  �  Z  %  �  �    o  ^  M  ?  4  *  $        �  �  �  �  �  �  �  s  �  <  S  d  i  [  N  j  i    �  r  L    �  V  �    I  &  �      "         �  �  �  x  ?  �  �  �  t  3  �  l  
   �            �  �  �  �  �  �  �  �  �  }  �  �  �  g  &  �  �  �  �    >  M  M  C  .    �  �  @  �  r  �  �  a   �  \  P  A  .    #      �  �  �  }  M    �  �  x  8  L  �  �  �  �  �  �  v  d  N  6    �  �  �  �  j  0  �  s    �    �  �  �  �  �  �  �  �  �  �  s  d  L  4  �  �  �  ?  �  �  �  z  s  m  e  V  H  9  +      �  �  �  �  �  �  �  �  �  �  y  h  Z  V  S  O  L  K  I  H  U  l  �  �  �  i  K  -  R  D  5  &      �  �  �  �  m  G  "  �  �  �  �  _  7    o  ^  M  <  +            �  �  �  �  �  �  �  �    i  �  �  �  �  �  �  s  Y  @  &    �  �  �  �  �  c  �  ;   �  �  �  �  �  �  y  q  g  \  R  K  G  D  O  y  �  �  �  �  �    �  �  �  �  �  z  V  ,    �  �  \    �  _  �  �    �  ~  e  M  6  /  %    �  x  I  (    �  �  �  �  �  �  �     �  �  �  �  �  w  U  /    �  �  j  "  �  b  �  s  �  p    �  $  �  L  �  �      �  �  W  �  5  u  t  �  (  C    �  )  +  *  %        �  �  �  �  �  `  0  �  �  T     �  V  G  9  +      �  �  �  �  i  F  #  �  �  �  g  .  �  �  x       1    �  �  �  �  �  �  ~  j  K  $  �  �  �  M    �