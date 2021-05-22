CDF       
      obs    /   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���"��`      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�*�      �  h   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       =      �  $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @D�          X  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vpQ��     X  '8   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�`          �  .�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >��      �  /�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�2   max       B$�j      �  0h   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�u,   max       B$�A      �  1$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�)�   max       C��      �  1�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�F�   max       C�Z      �  2�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         g      �  3X   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  4   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  4�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       Pj�      �  5�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�}�H˓   max       ?��]c�e�      �  6H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       >333      �  7   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @D�33333     X  7�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vk
=p��     X  ?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           `  Fp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�u�          �  F�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A      �  G�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�b��}W   max       ?�ۋ�q�     �  HH   
     g      0            s      >      2      !                  %      *               X   +   .   h   2      	            /   
   
   &   &   =   ?         N���N[B�P�ػNr�PArN	'Nf��N�SSP�*�N}~-P���NOJ�Pm�O��IP�N�/�OG�M���Nt�N�$O��"O��[O�`�N\IOV�Or�Op`�P�8O�vdPVнP9Pj�O�-4N�bN���N���O_'�O�=�N��>N���N��zOYR,O��O�s�N�'N8��O
�����C���o�T���49X�49X�#�
�#�
�t���`B;�o;��
<T��<�t�<�t�<��
<�1<�1<�9X<�`B<�h<�<�<��<��=C�=t�=�P=��=��=49X=@�=@�=D��=D��=y�#=y�#=}�=}�=}�=�%=��=��=�{=�E�=�Q�=[fgnit{������|ttgb[[<8:BNRTRNB<<<<<<<<<<#/BNg��������{[5
#%$#
�����
0FNOM:�����#&&#��������������������UPQT[grtzutmg[UUUUUU������5Nt�g[@)��������	 �����������)[jv��te[Q<.����������������������`]alw�������������z`%'5ACNgt�����t[NB5*%�������(5:9/������	""#"!	;2018<HKUanvngaXUH<;���

���������
$#$$







��������

�������������������������aaelt�������������pa�������
#/4;7/����������  ��������������������������������������������������GHNU]an�����zqnaUMIG)&#/9=9;Hmyz~}tmaH;)����������������!#)5B[ekf\F:)������6=DE8-'������)5BNB&� ��� �)6BKOHJNPJB=) ?9;@BNQUY[[[NB??????Y[bgt�������~tgg`[YY������������������������%'$ �������KEDOX[ht|{������t[Ktsrt}����������ttttt������������������������
 #%#
��������������������������������).20"����������
()#
��
 #+/145/#
rsnnz����|zrrrrrrrr--/4<HILNONKHF?<94/-�{ÇÉÓàæà×ÚÓÍÇ�}�z�n�n�k�n�z�{ŭŹ��������ŹŭŤŢŭŭŭŭŭŭŭŭŭŭ�)�6�O�tčġĥğčā�h�6�������������)�'�4�8�4�.�*�'����� �'�'�'�'�'�'�'�'�������ƻλλɻû��������x�F�&�/�?�_�x���ܻ�����������ܻѻܻܻܻܻܻܻܻܻܻ�ĚĦĳĽĿ��ĿĳĦģĚđĚĚĚĚĚĚĚĚ��������������������������������������������IŇřŠŗōŀ�[�<�-�"�����ľ������������߼ּʼ������Ǽʼּټ������;�F�,�&������s�M�5�(��4�M������𺗺������������������������������������������������ ��� �����������{�g�^�^�i���T�`�w���������������m�T�G�C�A�<�:�@�G�T�����A�N�U�K�@�5�(�����������ۿ����	���"�/�;�/�-�"���	� ������������������������
�������������������޺e�h�r�~����~�}�r�e�^�d�e�e�e�e�e�e�e�e��(�4�8�8�4�(���	�����������4�A�M�Z�f�h�s�{�|�s�f�Z�M�K�A�4�4�4�4�4ìù��������(�/�)���������úóñèì�4�A�M�Z�b�k�w�~�z�i�Z�A�4�(�����(�4�;�G�T�m�y�������������y�`�G�9�2�"��"�;���ʾ׾������׾ʾ��������������������5�A�N�Z�`�a�]�Z�N�A�2�"�������(�5�A�M�Z�d�c�`�b�f�g�f�Z�M�A�<�6�4�9�?�A�A�������������������������o�g�e�h�s������Ŀ�������
�#�0�7�7�%�������ĿĵĩĢĩĿ������*�2�=�F�O�C�6����������������ƁƧ������0�D�J�E�1�$�����Ƴƚ�}�d�qƁ�������ûٻ�����ٻܻ�ܻû����o�n�}��¿�����	�����¼�t�X�5� ���5�[¦²¿���!�3�N�P�P�F�:�!��������������H�T�a�m�p�p�m�a�T�H�H�G�>�E�H�H�H�H�H�H�/�4�/�.�.�$�"����	��	����"�-�/�/�	��"�'�+�)�%�"����	���������	�	�	���"�#� ���	������������������������'�4�L�Y�h�|�~�r�f�Y�O�@�4������������������������������������������������zÇÓàçèçàÓÇ��z�x�y�z�z�z�z�z�zDIDVDbDfDoDtDoDmDbD[DVDODKDID?D=DIDIDIDI�ùܹ���	��� �� ����Ϲù����������ú'�3�L�Y�r�������������~�r�Y�L�@�0�+�'�'E�E�E�E�E�E�E�E�E�E�E�E�E�E�ExErEsEuE�E�ǔǡǭǹǱǭǬǡǚǔǈ�{�t�v�{�}ǈǑǔǔ�����ʼּؼּܼϼʼ�����������������������������������������ŻŹŭŪŹ������ \ 5 ! T   R ] $ 6 n _ G " + N m G V @ S > , ; � B T % < " o 3 p > ; s 4 A = U % 5 ^ 5 Q P m g    �  Y  ?  G  (  )  �  �  �  �  S  g  �  =  �  �  d  9  }    E    �  �  �  w  �  �  �  p  �  �  �  �  �  �  �  �  �  �  �    6  u    �  v��1�D��>���49X=t��o��o:�o=���o=�+<#�
=�%=<j=L��<���=49X<���=\)=t�=�+=Y�=�hs=t�=D��=T��=P�`=��=��T=�1>�P=ě�=���=e`B=]/=�C�=�t�=�/=�t�=�hs=���=��>   >�P=�
==ȴ9>bNB	�B�sB	'B$��B$��B$�jB��B	3B�BE<BT�B"3�BzB��B��A�2B:�B$=�BB#��BALBCBBbBc'B 4B@B��A���B��BEBՆBϺBQ�B +B	�mBroB�+B�B�xB!��B� B��Bn�B��B��BY�B��B	��B |B	=\B$�AB$A]B$�~B�B	))BͷB:iBi�B")�B:�B	"B��A�u,B �B$?�B@@B#��B��B@
B?�BO�B>�B]�BA�A���B�MB9UB�B��B@'B>�B	�B��B9�B� B�B!�B�
BCYB@,B��BEMB?�B��Aɀ�A�pAٛ6@��.@���@�YA���A�A�a�AAN��@��A�I�Aj��A�-�A���A�n�?���A6A>g�A�u8A;$�Aj TAS�A��A=k\A��3A�WA��BD@��"A���@i/A�˧A�jzA���A��@͍�@��[A�8C�z>�)�?�C��BH@�k�A�`�Aɀ�A��}Aك�@��@��w@�txA���A��ZA�A�4AQc @яA�eAm2A��ZA��A�~�?��6A3UA=V=A�~A<��Ajw�ASKUA���A<�CA��`A���A�� B� @��A�zt@l"7A��eA�YVA��1A�?<@��@���A���C�yw>�F�?�ӁC�ZBB$@��A�^j   
     g      0            t      ?      3      !               	   &      *               X   ,   /   i   2      	            /      
   &   '   =   @      	            ?      1            C      M      1   %   +                  #   !   !               )   !   ;   +   ;   !               #               #   !                        )                  %      !   %   +                  #   !                        /      ;   !               #                           N���N[B�O�E]Nr�P8�N	'Nf��NkCO�U�N}~-O�z�NOJ�O��O��IP�N�/�N��AM���NH�iN�$O��"O��[On�;N\IOV�Or�O?;O.
�O�ޫP"M�O�*Pj�O�-4N�bN���N���O_'�O�=�NP�N���N���OP��O���O0��N�'N8��O
�  p  �  �  @  c  j  �  �  	  �  �  �    �  7  �  �  �  �  |  A  �  �  �  B    2  �  w  Y  g  �  �  &  �  �  2  �  U  �  
�  �  ,  �  R  �  ̼���C�>333�T���o�49X�#�
��`B=m�h��`B=��;��
=+<�t�<�t�<��
<�`B<�1<�j<�`B<�h<�=#�
<��<��=C�=��=��T=,1=<j=�O�=@�=@�=D��=D��=y�#=y�#=}�=�%=}�=�C�=�+=�{=��=�E�=�Q�=[fgnit{������|ttgb[[<8:BNRTRNB<<<<<<<<<<?<=@N[gt������tg[NF?
#%$#
����
#<DE?5-
�����#&&#��������������������ZSTX[egtttqgc[ZZZZZZ���)26874/)	�����	 ��������#)6BO[_dfhhf[OB7��������������������tppw�������������zt%'5ACNgt�����t[NB5*%�������(5:9/������	""#"!	646<HUW`[UH<66666666���

���������
!##��������

�������������������������aaelt�������������pa����
"/0461/#
��������  ��������������������������������������������������OKIQUanxz���zxnaXUOB==BHNTaemnqpmlaTMHB���������������	%)5B[ghb[G5)	�����%,,'#���������)5BNB&� ��� �)6BKOHJNPJB=) ?9;@BNQUY[[[NB??????Y[bgt�������~tgg`[YY������������������������%'$ �������KEDOX[ht|{������t[Ktrt~��������~ttttttt������������������������
"
��������������������������������� �������������
""
���
 #+/145/#
rsnnz����|zrrrrrrrr--/4<HILNONKHF?<94/-�{ÇÉÓàæà×ÚÓÍÇ�}�z�n�n�k�n�z�{ŭŹ��������ŹŭŤŢŭŭŭŭŭŭŭŭŭŭ�B�O�[�h�z��}�x�o�[�O�B�6�,�"�!�$�-�6�B�'�4�8�4�.�*�'����� �'�'�'�'�'�'�'�'�������ƻ»����������l�S�F�3�5�;�M�_�x���ܻ�����������ܻѻܻܻܻܻܻܻܻܻܻ�ĚĦĳĽĿ��ĿĳĦģĚđĚĚĚĚĚĚĚĚ�������������������������������������������
��#�:�I�M�K�=�0�#��
��������������������߼ּʼ������Ǽʼּټ����׾������������׾�������z������ʾ׺����������������������������������������������������������������������|�{�~�����T�`�w���������������m�T�G�C�A�<�:�@�G�T�����A�N�U�K�@�5�(�����������ۿ����	���"�/�;�/�-�"���	� �����������������������������������������������e�h�r�~����~�}�r�e�^�d�e�e�e�e�e�e�e�e��(�4�5�5�4�(���
�����������4�A�M�Z�f�h�s�{�|�s�f�Z�M�K�A�4�4�4�4�4ìù��������(�/�)���������úóñèì�4�A�M�Z�b�k�w�~�z�i�Z�A�4�(�����(�4�`�m�y�����������{�y�m�`�G�C�<�=�A�G�T�`���ʾ׾������׾ʾ��������������������5�A�N�Z�`�a�]�Z�N�A�2�"�������(�5�A�M�Z�d�c�`�b�f�g�f�Z�M�A�<�6�4�9�?�A�A���������������������������v�s�j�k�s�v������������������������������ĿļĿ����������*�6�B�F�G�6�����������������ƚƧ��������1�?�>�,��������ƜƊƂƈƚ�����ûлػ߻޻лû��������|�y�}��������¿�����	�����¼�t�X�5� ���5�[¦²¿���!�3�N�P�P�F�:�!��������������H�T�a�m�p�p�m�a�T�H�H�G�>�E�H�H�H�H�H�H�/�4�/�.�.�$�"����	��	����"�-�/�/�	��"�'�+�)�%�"����	���������	�	�	���"�#� ���	������������������������'�4�L�Y�h�|�~�r�f�Y�O�@�4������������������������������������������������zÇÓàçèçàÓÇ��z�x�y�z�z�z�z�z�zDIDVDaDbDoDrDoDjDbDVDNDIDADADIDIDIDIDIDI�ùϹ�����������ܹù������������úY�e�r�~�����������~�r�m�Y�L�>�8�7�@�N�YE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ExExEE�E�ǔǡǭǹǱǭǬǡǚǔǈ�{�t�v�{�}ǈǑǔǔ�����ʼּؼּܼϼʼ�����������������������������������������ŻŹŭŪŹ������ \ 5  T " R ] - 5 n N G  + N m - V I S > , - � B T    l  p > ; s 4 A = C % : Y % B P m g    �  Y  g  G  t  )  �    Y  �    g  �  =  �  �  �  9  a    E    �  �  �  w  �  l  }  k  e  �  �  �  �  �  �  �  m  �  �  �    �    �  v  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  p  \  I  2      �  �  �  �  �  x  �    i  R  6    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  v  9    ]  Z    N  S  %  �  �  �  q  �  /  �  H  �  �  �  @  <  8  4  0  ,  )  %  !                          ;  V  a  b  X  K  :  (    �  �  �  �  r  E    �  �  �  j  g  d  a  ^  Z  W  K  ;  *    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _    �  _  =     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  C  �  �    T  v  �  %  �  �  �  	  	  �  �  3  �  _  !  �  �  �  �  �  �  }  j  X  G  8  *        !  &  O  �  �  �  �  �  (  A  I  \  e  p  u  {  �  {  b  B    �  l  �  a  �  �  �  �  �  �  {  u  n  d  U  F  7  "  	  �  �  �  �  �  �  ;  k  �  �  �  �            �  �  �  �  M  �  �  0  �  �  �  �  �  �  �  �  �  �  �  �  |  i  S  :    �  �    �  7  2  %      	       3  0  "  	  �  �  �  L    �  A  �  �          �  �  �  �  �  r  R  2    �  �  �  }  W  0  �  �      3  �  �  �  �  �  q  Z  :    �  z  "  �  n    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          B    �  �  �  �  �  �  �  �  |  g  P  |  u  n  f  ^  U  L  C  9  .    
  �  �  �  �  b  J  Q  X  A  2      �  �  �  �  T    �  �  v  5  �  �  <  �  �  K  �  �  �  �  �  �  p  ]  K  8    �  �  �  x  P    �  |  �  |  b  K  @  1  �  �  }  Q    �  �  B  �  �  >  �  i  �  .  �  �    z  t  m  f  ^  V  N  F  >  5  ,  $        �  �  B  ?  <  8  /  !    �  �  �  �  z  L    �  �  N      �      �  �  �  �  �  �  �  �  �  m  ?    �  u    �  5  �  $  .  1  /  (      �  �  �  �  �  r  N  $  �  �  �  L    h  �  	K  	�  
d  
�  #  m  �  �  �  �  d  
�  
}  	�  �  �  q  1  l  u  v  m  Z  A    �  �  �  J  	  �  q    �  6  �  �  �  E  N  A  X  R  9    �  �  �  |  P    �  x    �  P  |  e  	�  
�    K  c  e  [  F  (  
�  
�  
y  
  	�  	6  }  �      �  �  �  �  �  �  ~  n  �  �  �  �  �  w  D    �  (  �  =  �  �  �    a  B    �  �  �  G  �  �  >  �  �  Q    �  ]  �  &      �  �  �  �  �  �  �  �  y  c  N  9  /  *  #      �  �  �    �  �  �  �  �  �  �  �  �  �  w  j  g  �  �    �  �  �  w  k  ^  P  B  3  "    �  �  �  �  �  �  �  �  �  2  2  /  #      �  �  �  �  �  �  d  :    �  �  �  G    �  �  �  �  w  8  �  �  	    )  6  7    �  @  �  "  �  *  5  F  S  E  5      �  �  �  �  ]  :    �  �  {  C    �  �  �  �  �  �  �  �  �  �  {  s  j  b  W  G  1    �  r    
�  
�  
�  
�  
�  
�  
�  
z  
J  	�  	�  	  h  �  2  �  �  7  T  M  �  �  �  �  �  :  �  �  =  O    �  �  [    �  �  A    �  �          '  +  (    �  �  p    �  -  �  �  L  f  B  �  �  �  �  �  �  �  �  �  M  
�  
o  	�  	^  �  �  O  �  �  O  R    �  �  �  �  �  j  C    �  �  �  P    �  �  �  Z  +  �  �  �  �  w  f  R  >  2  '         �  �  �  �  �  �  �  �  �  �  �  �  }  \  7    �  �  �  Y    �  �  E     �  �