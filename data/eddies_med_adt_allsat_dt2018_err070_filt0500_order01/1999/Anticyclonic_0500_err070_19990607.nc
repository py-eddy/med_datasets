CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�?|�hs      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Q�      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �u   max       =�"�      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E���Q�            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vs33334        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O            h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >aG�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��0   max       B,�L      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~<   max       B,~�      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C��@      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Ȑ�   max       C���      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          I      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��3���   max       ?����#�      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �u   max       >J      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E���Q�        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vs33334        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O            h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?����#�     @  L�                  -      9   6         �                        '   �               6            F      
      0                        
   R         9      �         1   Nt��O��aQ�N6ӊN�Z3O��sN�[P���PZUO	�7N��PÉ�N���O��>N���N|O��{N=OI�P's=P5C�O`1O6_�Og�N�|PA�O ��O��UO7*�P�QN<c�NQ$mN0}O�ebO�K�Odl�N���O5|�O"��N}�5OfwN��rP�LM��N(y2O��O�eO���N<��Nτ�O�|N��A�u�e`B��o;o;o;D��<o<#�
<49X<49X<49X<D��<D��<e`B<�o<�o<�t�<�1<�1<�j<ě�<���<���<�/<�/<�h<�h<�h<�h<�=o=C�=C�=�P=��=��=�w='�='�='�=8Q�=D��=D��=P�`=aG�=�+=�+=�\)=�t�=��T=�{=�"�#,02650,# 	"&/4;?DMJH;/"����Thnsrg[B4 �����������������������0,(01<IRNII<10000000g`]co�������������tg!#)/<>HJKHD</%##!!!!�������AC/! �����5BLTNNRQLB5) 
�����
#
�������jddfnz��������znjjjj��������"8@?B>5�����������������������)5BN[bkprpg[N5)����������������������������������������]^gt������������}tg]VUanqz�znaVVVVVVVVVVksz�������������zqnkAA<?Taz��������zaTHA<9BN[g����������g[N<����������������������������������������  
/<ADC<#
YSX[][Y[hotuyvtoh^[Ykinz�������������zuk��������������������������

����������
 #$&%#"
��������������������������������������������TUZaenutnaVUTTTTTTTT������������������������
/;B;)#
	����������������������������$**'����������������������������������#&)/6BO[bhjnh`O>6)R[ahnt����{th[RRRRRR#/4<AGMMKH<7/+(&%#216>BGOV[ha^[SODB622?BGW[h{�������thOJF?��������������������#$/<C<</#����������������)58BQ[iqgN5����������
!$!
������������������������������ ����������������

�������������������������Ľнٽݽ�ݽؽнĽ����������ĽĽĽĽĽ��T�a�g�m�y�z���z�u�m�T�;�/�,�'�$�,�;�H�TƳ������H�F�����Ǝ�u�6����C�O�u�}Ƴ�U�a�k�c�a�U�H�@�H�J�U�U�U�U�U�U�U�U�U�U�f�r�������������r�r�h�f�e�f�f�f�f�f�f����������������
����������������������������������������������������������������g¿½¬¦�p�g�B�*���������������
�/�HÒÚÓ�z�H�/�������­¢¿�������
�m�t�z�}�}�|�z�x�m�a�X�T�O�N�R�T�a�a�m�m���������������������������������������������<�bŠ������ŠŇ�n�U�������Ĺ������ɺкɺǺ����������������������������ɺ����������������������������������������޻��!�-�8�:�:�<�:�-�!���������������ʼּ������ּʼ��������������������T�a�h�l�q�m�i�a�\�T�G�;�7�5�7�B�H�M�P�T�a�n�u�s�n�b�a�a�]�Z�a�a�a�a�a�a�a�a�a�a��������������� �	�������������������������
�#�0�I�K�C�9�#�����������������������)�B�Q�\�c�d�[�O�?�6�)��������������4�A�M�Z�_�f�h�Z�X�M�A�7�������(�4������(�/�,�(�(�#��������޿����g�s�����������������������������~�s�_�g�/�;�H�T�a�i�m�n�j�a�N�H�;�/�+�"��"�'�/���������лڻ߻�ܻл��x�^�T�T�_�p�������4�A�M�V�W�Z�Z�Z�X�M�M�A�4�0�(�#�!�(�/�4�ʾ׾߾������׾ʾ������������������ʾN�Z�f�v������s�k�f�Z�M�C�A�3�/�4�A�L�N�������&�&�!�������ëáá×Øàìù���B�O�W�[�g�[�O�B�7�<�B�B�B�B�B�B�B�B�B�BE�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E򿫿��������������������������������������G�T�m���������y�`�G�.�����ھ��	�"�4�G�f�s�v������z�q�d�Z�Y�M�;�4�,�)�2�A�M�f�f�s�����������������s�f�_�\�]�a�`�_�fù����������üùìàÓÓÓÖÛàìðùù��!�-�:�;�D�B�:�6�-�!������������!�-�:�=�F�K�O�S�Z�_�b�_�S�F�:�,�!���!�@�F�L�P�Y�]�\�Y�L�F�@�:�3�;�@�@�@�@�@�@������������s�f�Z�M�I�@�A�E�M�Z�f�s���������żʼ˼ʼż��������������������������4�Y�f������r�Y�M�@�4�����������4�ûлܻԻлû��������ûûûûûûûûû�����������������������������������������/�H�a�m�����������z�m�a�T�=�"�������������̿ѿֿܿԿѿ��������y�o�f�k�y����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDxD��y��������������y�t�q�s�y�y�y�y�y�y�y�y�/�<�H�O�U�Z�`�U�H�<�/�,�&�+�/�/�/�/�/�/EuE�E�E�E�E�E�E�E�E�E�E�E�E~EuErEpEoElEu�ּ�����������׼ּӼּּּּּּּ� 8 8 > f C D # f o A ; P Y d E | v i M ( ! F 5 Y Y F / / - B I 7 = s : > 3 & u o \ [ E O | m Q  Q , 2 +  �  0  	J  8  �  �  �  �  p  F      .  �    }  �  |  �  �    �    C  *  �    U  }  s  e  d  P  �    �  �  v  �  �  n  �  �    �  �  J  �  [  �  \  ��#�
;�`B>%;ě�<t�=D��<49X=�7L=��<�9X<�t�>\)=\)=0 �=+<�9X<�<ě�=t�=}�>A�7=t�=�P=<j=C�=��=#�
=8Q�=<j=���=0 �=49X=t�=�1=�C�=y�#=m�h=�+=u=Y�=ix�=ix�>�=ix�=}�=��#=ě�>aG�=��-=��>1'>JB%Q%A��0B��B��B&=B
�"B~�B;�B7(BZLBV=BkB!�;BĉB QtB!E|B.�B�hB�<A���B	�@B�;B^�B�B��B!B  B#��B$l�B�uB��BV�B`}Bs�B�B��B!��BuBRB�oBooB��B�}B#�B
B�/BLVB1B,�LB��B.�B`[B%@CA�~<B?�B��B&4�B
��B��BBB�
B��B��BAeB!�kB��B ?�B!?�B8`B�B(�A��'B	�VB6�B�xB�B�
B>zB�9B#H�B$�B��B@�B@B>3BA�BI�B�lB!��B@XB@/B@�B�sB>�B� B#B�B}�B��B><B,~�B�BM_BE�A'�wA� �BޱAŘ�@���A�1�A�طA��QA¯A��A���A�q@��A���@gr=A �A���AƟIA�e*A��dA� �A9!�A�0A�4zA�W�@��A:�iAP�tA>��A�ԚA�R�C��@At��AiT�A>�/AC�zA̦�@i��@ze?��AA�D@�B#@��@��A���A���Ar�C���A]PA��|C�AN$A(�7A�~�B��Aģ�@���A���A�~qA�S�Aª�A��bA���A궧@��A�u�@k�(@��AA���Aƒ�A�~2A�qA�p�A7MA��A���A��@���A9��AP�A>��A�~?A؀�C���At�Aj�A?~AD�yA�h@k�>@|y?Ȑ�AB��@�h�@ͣo@��tA��A�m�As1�C��A�A��C��A�
         �         .      :   7         �                        (   �               6            G            0                        
   R         9      �         1            O               I   9         C                        '   )               )            #            +                           )         -   %                        I               '            1                        !                                                                              !   %               Nt��N�InP��N6ӊN�Z3Op�N�[P��O�dJO	�7N��Pm�N�_�O���N���N|O��{N=N��O�+�O�WO`1O6_�Og�N�|O���O ��O��UN�_�O�TjN<c�NQ$mN0}OW�WO_r�Odl�N��kN��	N��IN}�5OfwN��rOu!M��N(y2O�J�O�rXO2��N<��Nτ�O�|N��A  �  %  �  -  R  G  �  E  
  �  +  	�    W  �  �  1  �  �  �    M  �  �  O    �  �  �  
�  W  �  �  �  �  �  R  n  �  �  �  �  �  C  �  
  �  �  �  �  n  3�u���
<D��;o;o<�C�<o=o=\)<49X<49X=0 �<�C�<u<�o<�o<�t�<�1<ě�<�=�9X<���<���<�/<�/=49X<�h<�h=+=D��=o=C�=C�=L��='�=��='�=<j=<j='�=8Q�=D��=�^5=P�`=aG�=��P=�7L>J=�t�=��T=�{=�"�#,02650,#"/4:9//"����5Udimj[NB5�����������������������0,(01<IRNII<10000000eglt�������������the!#)/<>HJKHD</%##!!!!�����������'&
���)58ACEDB>5)�����
#
�������jddfnz��������znjjjj�����$+15861)�������������������������)5BN[akpqog[N5)����������������������������������������]^gt������������}tg]VUanqz�znaVVVVVVVVVVwnwz~������������zwHIJMamz��������zaPKHKLO[gt��������tg[UOK����������������������������������������  
/<ADC<#
YSX[][Y[hotuyvtoh^[Y���������������������������������������������

����������
!#$##
��������������������������������������������TUZaenutnaVUTTTTTTTT�����������������������
#/5:;71/#
����������������������������$**'������������������������������ ����2--36BO[][[OJB862222R[ahnt����{th[RRRRRR#/4<AGMMKH<7/+(&%#216>BGOV[ha^[SODB622TOOS[htw���}xtkhb[T��������������������#$/<C<</#����������������� )57AP[hpg[NB5��������

������������������������������� ����������������

�������������������������Ľнٽݽ�ݽؽнĽ����������ĽĽĽĽĽ��T�a�i�m�o�q�m�a�T�H�G�A�H�M�T�T�T�T�T�T��������4�7�(�����Ɓ�h�1� ��)�U�uƎ���U�a�k�c�a�U�H�@�H�J�U�U�U�U�U�U�U�U�U�U�f�r�������������r�r�h�f�e�f�f�f�f�f�f�������������� �������������������������������������������������������������������������)�N�[�n�s�o�[�N�B������������/�<�H�U�g�p�n�a�U�<���������
���#�/�m�t�z�}�}�|�z�x�m�a�X�T�O�N�R�T�a�a�m�m�����������������������������������������#�I�{ŊōŁ�n�b�I�0�����������������
�#�������������������������������������������������������������������������������޻��!�-�8�:�:�<�:�-�!���������������ʼּ������ּʼ��������������������T�a�h�l�q�m�i�a�\�T�G�;�7�5�7�B�H�M�P�T�a�n�u�s�n�b�a�a�]�Z�a�a�a�a�a�a�a�a�a�a�������������������������������������������
�#�0�=�@�;�2�#�����������������������)�6�I�R�S�O�G�6�)���������������4�A�M�Z�_�f�h�Z�X�M�A�7�������(�4������(�/�,�(�(�#��������޿����g�s�����������������������������~�s�_�g�/�;�H�T�a�i�m�n�j�a�N�H�;�/�+�"��"�'�/�������ûлٻۻۻлû���������x�t�x�����4�A�M�V�W�Z�Z�Z�X�M�M�A�4�0�(�#�!�(�/�4�ʾ׾߾������׾ʾ������������������ʾM�Z�c�f�s�{�z�s�f�\�Z�O�M�A�8�5�A�B�M�M����������������������ùîíëù���B�O�W�[�g�[�O�B�7�<�B�B�B�B�B�B�B�B�B�BE�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E򿫿��������������������������������������m�u�y��������y�m�`�T�G�D�7�:�G�J�T�`�m�f�s�~��x�o�f�a�Z�M�>�4�0�.�4�9�A�M�Z�f�f�s�����������������s�f�_�\�]�a�`�_�fàìù��������ûùìàÕØÝàààààà��!�-�4�:�?�<�:�.�-�!�������������!�-�:�F�H�L�M�K�F�:�:�1�-�(�!��!�!�!�!�@�F�L�P�Y�]�\�Y�L�F�@�:�3�;�@�@�@�@�@�@������������s�f�Z�M�I�@�A�E�M�Z�f�s���������żʼ˼ʼż���������������������������'�4�@�M�U�P�M�@�:�4�1�'��������ûлܻԻлû��������ûûûûûûûûû����������������������������������������H�a�m�|�����������z�m�a�T�H�@�.�'�/�1�H�������ʿ׿ӿѿ��������y�p�j�g�j�l�z����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��y��������������y�t�q�s�y�y�y�y�y�y�y�y�/�<�H�O�U�Z�`�U�H�<�/�,�&�+�/�/�/�/�/�/EuE�E�E�E�E�E�E�E�E�E�E�E�E~EuErEpEoElEu�ּ�����������׼ּӼּּּּּּּ� 8 & = f C B # [ S A ; ? ; b E | v i 8 #  F 5 Y Y ? / / ( ? I 7 =  = > . + Y o \ [  O | B R  Q , 2 +  �  �  �  8  �    �  �  �  F    �  �  x    }  �  |      Z  �    C  *  K    U     o  e  d  P  �  �  �  �    �  �  n  �  A    �  �  )  s  [  �  \  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �  �  �  �  �  �  �  �  {  r  `  I  1        �   �   �  �      �  �  �  �  �           �  �  �  P  �  `  �    9  �  �  �  h  V  1  �  �  �  t  t  �  �  �  y  �  ?  K    -  %          �  �  �  �  �  �  �  �  �  �  �  �  �  �  R  O  M  I  E  A  <  6  0       �  �  �  �  �  �  q  X  @  �    '  6  ?  E  G  B  )    �  �  n  )  �  P  �  =  �  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  R  A  1  �     �  0  ?  8  "  <  B  9    �  �  e     �  R  �  ;    D  v  x  �  �  �  �  �  �    �  �  �  �  I  �  �  �    �  �  �  �  �  �  �  �  �  r  X  ;      �  �  �  �  �  v  F  +  !      �  �  �  �  �  �  r  W  9      �  �  �  �  �  q  �  	d  	�  	�  	�  	�  	s  	^  	S  	N  	>  	0  	  �  B  Y    �  :  �  �  �  �     �  �  �  �  �  z  G    �  �  )  �    �    P  S  C  (    �  �  �  �  m  A    �  �  v  ,  �  c      �  �  �  �  �  �  �  �  �  �  v  S    �  n    �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  `  O  2    �  1  !    �  �  �  �  {  T  4    �  �  �  g  >      �  �  �  �  �  �  �  �  �  �      
        9  T  p  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  k  x  �  �  �  �  �  �  �  �  �  �  q  d  M  7    �  �  �    !   h    1  �  �  !  �  �    �  �  �  -  �  �  �  �  J  	�  �  �  M  7       �  �  �  �  e  ?    �  �  �  �  �  �  �  h  F  �  �  �  �  �  �  �  �  �  {  c  N  :  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  M  +    �  �  g  ?    �  O  O  N  I  7  %    �  �  �  �  �  �  �  o  Q  2     �   �  �  �  �  �  �        �  �  �  �  �  b  "  �  d  �  2  %  �  �  �  �  �  �  �  �  �  �  �  �  �  p  X  A  (    �  �  �  �  �  �  �  �  y  i  X  G  5  #    �  �  �  �  �  �  s    m  }  �  �  �  �  �  x  e  P  6    �  �  �  �  \  +  �  u  	T  	�  
!  
X  
z  
�  
y  
R  
  	�  	�  	$  �  1  �    Y  f  :  �  W  [  b  r  }  �  �  �  �  �  �  �  �  �  �    Y  �  �    �  y  m  X  A  &    �  �  �  �  �  }  _  7    �  �  K  	  �  �  �  �  �  }  y  t  p  k  g  b  ^  Z  U  Q  L  H  D  ?  {  �  �  �  �  �  �  �  m  C    �  |  )  �  ]  �  U  �  �  �  �  �  �  �  �  t  R  /    �  �  N  �  �    �  4  �    �  �  �  s  _  N  4    �  �  J  �  �  [     �  C  �  �    H  F  R  N  F  ?  0      �  �  w  @  �  �  i  C  �  D  �  E  U  b  l  k  Z  A  %    �  �  �  X  !  �  n  �  �       �  �  �  �  �  �  �  �  �  �  `  0    �  �  8  �  C  �  8  �  �  p  T  ;  !    �  �  �  Q    9  ;  �  �  '  �  a  �  �  �  �  �  �  v  n  T  0    �  �  �  �  �  �  n  a  |  �  �  v  c  M  7  !    �  �  �  �  �  �  �  �  �  �  �  �  �    O  N  E  4      0  c  �  �  �  b  )  �  z  �  �  �    C  @  =  :  5  /  )  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  y  z  b  G  *  
  �  �  �  v  .  �  �  �  	:  	G  

  	�  	�  	o  	'  �  �  2  �  �  &  �  .  �  �  �  �  �  �  �  �  f  8    �  �  �  �  e  4    �  �  b    w  �  �  b  �    5  ^  v  �  r  I    �    d  �  Q  �  J  
z  =  �  �  �  �  �  �  �  �  �  �  �  �    s  g  [  :    �  �  �  �  �  u  O  +    �  �  �  �  ^  >  $  	  �  �  J  �  �  n  e  U  B  *  �  �  0  �  Z  
�  
o  	�  	s  �  <  �  �  ?    3  "    �  �  �  �  {  M    �  �  �  M    �  ]  �  &  u