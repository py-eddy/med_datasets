CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Լj~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��4   max       P��j      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       >���      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E��
=q     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vr�\(��     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P@           d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @��          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       :�o   max       >���      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�R_   max       B-HX      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~%   max       B-N�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�&�   max       C�w�      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >䈲   max       C�xe      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         D      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��4   max       P���      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�D   max       ?�@N���U      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >���      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?G�z�H   max       @E�=p��
     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����T    max       @vr�\(��     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @N�           d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�3�          �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?�?|�hs     P  J  D   %         `                                    2                  =      /         ]   	   @            W   )                     I   	   Q               )P�KIPki\OWJO���P��jN�N,��N8�wNeN�xNJ�lN7ԡP	7�O���Nu O���P�)�N�NHN,Nx�yP��O{��P��7N/��O���O��ZN�tO�-N�fOPl�SO�pM��4NI$O�cO��N�OB�aO�z�O���N���N�_�P
WNa��O��N�b�N-��N�i�N���N����9X��1����D���ě���o%@  :�o;�o<o<49X<D��<T��<�C�<���<���<��
<�1<�1<�9X<�9X<�j<��<��=+=C�=C�=C�=\)=0 �=0 �=49X=@�=D��=D��=H�9=P�`=P�`=T��=e`B=e`B=u=���=��w=��
=��
=� �>��>���$9N[���������t[5$���"<bonUI98#
������������������������UQPO[gt��������tga[U�����N[tztRI<)������������������������������������������������������uxxz�����zuuuuuuuuuu�����

 �����������������������������77<HMTOHA<7777777777&*BJgt��������tcNB)&������
 ((& 
������eegkt������utgeeeeee
#0<IXaca[SF0%

��Ohru�����hB)647<HU`alfaZUHF<6666`[X`acnnrtna````````�������������������������
<GLH</���������
#+/688/#
��z������(1%�������~z����������������������������������������xppw}��������������x��)./)����������������������������{nbnt{�����������������)5DOPB5�����������������������nlkinrvz{znnnnnnnnnnNFGN[[_][NNNNNNNNNNN#'(/5;HTamswxvoaTH/#������ �������������������������������������  �������������$$
�������#/<JU`afa_O</#��������������������)69;<?6-)����)6BO[c\BA6)����}xx���������}}}}}}}}�������� 
#'(#
��WRSW[hptutphh[WWWWWWVU[[honkh[VVVVVVVVVVHHC</-,(-/<HHMHHHHHH;85447<AGHKLKIH<;;;;//<BC=</##/////�)�B�[čĠĢĘĆ��B�>�0�)�����������)�����һ׻û»������p�n�S�F�#�!�*�?�_�x��ĚĦĿ������������ĿļĳĦĚčĊċčĕĚ�����������������������������������������I�bŇūŠŒŋń�Y�<�#�
������Ļ�����#�I���������������ĿƿĿ�����������������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Ƴ������������ƳƨƮƳƳƳƳƳƳƳƳƳƳ�a�d�k�n�s�x�n�a�W�U�H�E�H�U�Y�[�a�a�a�a�ּ����ټּмʼļʼ˼ּּּּּּּ�D�EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�D��T�`�y���������������m�`�G�A�@�;�8�;�D�T�M�T�Z�f�k�u�x�s�f�M�A�4�(����$�A�E�M�A�M�Q�Z�b�b�e�Z�W�M�J�A�@�<�A�A�A�A�A�A����������������������r�f�^�f�r�{��������"�.�6�,����׾������M�J�S�M�|�����׾���������
���������������������������g�s�������������s�g�]�_�g�g�g�g�g�g�g�g���Ŀ̿ѿܿݿ޿ݿܿѿĿ�������������������5�C�I�K�C�2�-�(�������������m�y�����������y�r�`�T�G�<�:�;�@�J�T�\�m�����	�/�E�J�G�7��������u�n�������������A�N�R�X�Z�d�Z�N�K�A�@�;�A�A�A�A�A�A�A�Aù�����������������þðíîéìù���������������������������������u�u�}���f�h�k�q�n�j�f�Z�V�S�W�Z�^�f�f�f�f�f�f�f���Ϲ����!�'�"�����Ϲ����������������������������������������������!������²�g�S�L�0�'�5�N¡²�˾4�A�M�Z�Z�^�]�\�Z�P�M�I�A�4�0�(�&�)�-�4�n�zÇÓÙÓÇ�~�z�z�n�n�n�n�n�n�n�n�n�n�;�H�T�Y�T�T�H�;�:�;�;�;�;�;�;�;�;�;�;�;Ŀ��������'�3�,�"��
��������ĿĺķĵĿ�������Ļλһۻٻлǻû�������������������������������������������������������������*�-�6�8�?�C�G�C�6�*�����������l�o�y���������y�o�l�`�\�S�K�J�K�M�U�k�l�Z�f�s�|�x�n�Z�E�4�,�(�&������(�4�Z���	����"�*�/�1�/�"���	�� ���������!�)�-�3�4�9�8�-�!�����������!�!�r�������������������������j�Z�I�J�P�Y�r����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EuEtExE���'�4�@�K�I�@�6�4�2�'����������Y�f�r�s�v�r�f�Y�Q�O�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y¦¥¦²¿����¿¾²¦¦¦¦¦¦Ź�����������������������ŹŲŴŹŹŹŹǭǭǩǡǔǈǆǂǃǈǊǔǡǡǪǭǭǭǭǭ  =  , ' F H I N Y 9 / A 4 e , m 1 b W V L + p $ $ � K s a 3 x 5 / : r 4 d F z U ^ @ I 4 ; 5 x 9    }  �  �    �  $  P  [  +  �  b  E  �  �  �  O  \    Y  �  �    �  y  #  �      �  �  X  @  "  #  Q  Q  �  �  �  �  G  �  �  �  �  C  �  �  �>��<u;ě�<o=�E�:�o;ě�<49X;ě�<�t�<���<���=#�
='�<�9X=@�=�O�=0 �<ě�<�`B=Y�=8Q�=�j=t�=���=q��=#�
>%=0 �=�"�=�O�=@�=P�`>	7L=�9X=]/=�C�=��P=���=u=�%>+=�1>"��=�Q�=�{=���>(��>���B	G9B$�WB�B	��B��B0B�NB�B  �BM�B�B0B	�B#�VB	�B&#wBhvB=�Bx�B��B��BKOB1�B*�B>B�B��B��B)�B�B	eB�+Bo�A�R_B�B"1uBGB-HXB��B�XB��B��BmB��B|!B��B��B�MB>B	?�B$@B>vB	J�B�qBA�B�B�B 8GBB�B��B=�B	INB#��B	��B&�BXB?�B@�BՎBF�B9�B��B6&Bx�BA�B��B�\B(�}B��B<B��Bk�A�~%B�eB"=tB�kB-N�B��B��Bz�B�B�0B��BGTB�=B��B��B?{A��`@��A�&{A��^A�-�At$�A��C�w�B��AƼ�A4<C�L8Aj�ZA;� A=g@�Y
AM�\A�n#A��AyeA��Aj��A�zwA��XA�݃A���A@��>�&�@Z�gA�,�A;�A�'A���A��@�_�A qnA�U�A�]A;iA�SK@i�*@:9@�iC�%�@�r�@�%A�VA��BKqA�m�@��A�xPA��=A�At_JA�	�C�xeBZ�AƌcA �C�HAm�A;��A=�@���AQe�AҀ]A�%AyV�A���AjȑA�fmA��A��A�m�A@�>䈲@S�A��A;UA�p�A���A噍@���A!A���Au�A:��A���@k��@�k@���C��@�@��A���A��B@W  D   %         `                  	                  2                   >      /         ]   	   A            W   )                     I   
   R               *   ?   9         E                        '   !         A            )      E      #         %      =            !               !         -      !                     )         #                        '            +            %      9                     /                                    !                     O���P��O=#�OG��O�h�N�N,��N8�wNeN��NJ�lN7ԡP	7�O��Nu O�=;PNN@�JNHN,Nx�yO��uO{��P���N/��O�g�O��ZNDtoO��qN�fOP6
�O��M��4NI$OB9�O�)N�O3s�Oq�mO�HN���N�_�O���Na��O`�SN���N-��N�i�N���N��  �  �  D    3    "    �  �  �  �  �  �  Y  �  {    �  �  S  #  F  �  �    �  /  m  H  �  @  G  {  �  �  �  }  �  �  h  �    �  �  t  )  �  N>���#�
��C��o=L�ͻ�o%@  :�o;�o<49X<49X<D��<T��<�1<���<�9X=t�<�h<�1<�9X<���<�j=t�<��=L��=C�=\)=m�h=\)=T��=8Q�=49X=@�=���=H�9=H�9=T��=Y�=Y�=e`B=e`B=�t�=���=Ƨ�=��T=��
=� �>��>���ABDN[gt�������tg[NDA�����#<IIE</.#
�����������������������ZTST[gtv��������tg[Z����'*+)*'$�����������������������������������������������������uxxz�����zuuuuuuuuuu����

��������������������������������77<HMTOHA<7777777777&*BJgt��������tcNB)&������
#&#������eegkt������utgeeeeee#0<IT^`^UOIA0)!)6O[fjlrx~{thO4;9<<HPU\URH<;;;;;;;;`[X`acnnrtna````````�������������������������
<EE</#�����������
#+/688/#
���������)/*"�����������������������������������������������xppw}��������������x�)+,)	�����������������������������{nbnt{����������������)5ALMA �����������������������nlkinrvz{znnnnnnnnnnNFGN[[_][NNNNNNNNNNN988;<CHTahmnmjaTHB;9������������������������������������������������������������������#/<IU_`]UM</+#��������������������)69;<?6-)������)/5462)����}xx���������}}}}}}}}�������
! 
����XSTX[hnsohf[XXXXXXXXVU[[honkh[VVVVVVVVVVHHC</-,(-/<HHMHHHHHH<86558<@FHJKJHB<<<<<//<BC=</##/////�B�O�[�o�x�y�u�h�[�O�B�6�)�����#�6�B���������ƻ������������x�S�F�:�3�=�_�x��ĚĦĳĿ������������ĿĳĦĚĐčččĘĚ�����������������������������������������#�<�I�R�_�a�Z�I�<�0�#���������������#���������������ĿƿĿ�����������������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Ƴ������������ƳƨƮƳƳƳƳƳƳƳƳƳƳ�a�a�n�p�q�n�a�]�U�S�U�a�a�a�a�a�a�a�a�a�ּ����ټּмʼļʼ˼ּּּּּּּ�D�EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�D��T�`�y���������������m�`�G�A�@�;�8�;�D�T�A�M�Z�a�f�p�q�l�f�S�A�3�(�%����(�0�A�A�M�Q�Z�b�b�e�Z�W�M�J�A�@�<�A�A�A�A�A�A������������������������{�r�l�f�a�f�r����׾��	������׾�������s�i�l�������������������������������������������g�s�������������s�g�]�_�g�g�g�g�g�g�g�g���Ŀ̿ѿܿݿ޿ݿܿѿĿ�������������������(�/�D�E�>�/�(����������������m�y�����������y�r�`�T�G�<�:�;�@�J�T�\�m����/�<�A�G�D�3�#����������������������A�N�R�X�Z�d�Z�N�K�A�@�;�A�A�A�A�A�A�A�A������������
�������������üøûþ�����������������������������������u�u�}���f�h�o�m�h�f�Z�X�U�X�Z�d�f�f�f�f�f�f�f�f�ùϹܹ�����������Ϲ������������ú�������������������������¿����� ��������²�t�[�B�E�Nª´¿�4�A�M�W�Z�]�\�[�Z�M�A�4�1�(�(�(�*�0�4�4�n�zÇÓÙÓÇ�~�z�z�n�n�n�n�n�n�n�n�n�n�;�H�T�Y�T�T�H�;�:�;�;�;�;�;�;�;�;�;�;�;�����������
���
���������������������ػ������Ļλѻֻڻػлû�������������������������������������������������������������*�,�6�=�@�6�*����������������`�l�y�����������y�u�l�`�Y�S�M�K�L�O�V�`�Z�f�s�z�w�m�Z�C�4�(��������(�4�Z���	����"�*�/�1�/�"���	�� ���������!�)�-�3�4�9�8�-�!�����������!�!�r�~���������������������~�r�e�P�Q�W�e�r����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEyE}E�E�E���'�4�@�I�F�@�4�'������������Y�f�r�s�v�r�f�Y�Q�O�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y¦¥¦²¿����¿¾²¦¦¦¦¦¦Ź����������� ������������źŹŸŹŹŹŹǭǭǩǡǔǈǆǂǃǈǊǔǡǡǪǭǭǭǭǭ  5  + I F H I N B 9 / A + e % a 5 b W W L % p ! $ r 9 s ` . x 5  9 r * \ F z U " @ < 6 ; 5 t 9    �  l  �  �  �  $  P  [  +  1  b  E  �  A  �       Z  Y  �  �    w  y    �  �    �  a  .  @  "  �  1  Q    -  �  �  G  �  �  �  �  C  �  �  �  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  	�  �      �    L  =  �  N    R  �  ^  �  )  *  �  �         2  D  c    �  �  i  J  #  �  �  �  �  �  �  �  T    :  B  C  B  =  4  '    �  �  �  |  >  �  �  B    �  J  �                �  �  �  �  �  �  c  C  !  �  �  p  �  C  |  �  �  �  �  �  y  �  �    2  &  �  �  Y    W  x  �        �  �  �  �  �  �  �  �  �  w  h  Z  K  -     �   �  "  "  #  $             �  �  �  �  �  �  �  �  �  u  d      �  �  �  �  �  �  �  l  A    �  �  }  Q  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �           *  3  =  P  _  n  {  �  �  �  �  �  �  �  �  �  �  �  �  r  X  ;    �  v  l  ^  O  ?  .      �  �  �  �  p  @    �  �  r  =  �  {  [  :    �  �  �  ~  X  0    �  �  O    �  e    �  �  �  s  T  3    	        $  !    	  *  1    �  @  �  �  �  �  �  �  �  �  �  �  y  S  )  �  �  �  M  �  �  [  5  Y  Y  X  X  W  W  W  M  >  /  !       �   �   �   �   �   w   _  ]  �  �  �  �  �  p  X  <    �  �  q  6  �  �  �  ]  �  �  K  T  Y  [  S  y  {  q  U  ,  �  �  a  �  p  ?    �  *  �  |  �  �  �  �  �        �  �  d    �  )  �  :  �  6   �  �  �  �  �  �  �    z  t  o  i  c  ]  W  P  J  C  =  6  /  �  �  �  �  �  �  �  �  ~  n  \  K  9  '       �  �  �  �  G  N  R  S  J  >  6  7  '      �  �  q  ,  �  �  P  �   �  #      	  �  �  �  �  x  K    �  �  k  -  �  �  �  S    9  C  @  >  4  &      �  �  �  A  �  �  E  �  J  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  v  ]  C  *    �  �  �  s  �  �  �  �  �  �  �  �  �  �  _    �  =  �  j  �  �  %                �  �  �  �  |  T  +    �  �  u    a  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  ~  }  z  j  Y  I  
�  H  �  �    .  &    �  �  �  F  
�  
r  	�  	1  j  �  c  �  m  [  J  7  %    �  �  �  �  �  �  �  �  �  w  l  c  Z  Q  �    ;  H  D  7    �  �  �  �  �  Q     �  9  �  �  !  c  �  �  �  �  �  h  F    �  �  w  :  �  �  B  �  �  n  1  �  @  C  G  K  O  S  W  [  ^  `  c  f  i  n  u  {  �  �  �  �  G  B  =  9  4  /  )  #          �  �  �  �  �  �  h  H  	^  
  
�  M  �    R  r  z  `  (  �  J  
�  
;  	�  q  �  Q  �  �  �  �  �  �  n  G    �  �  L  �  �  #  �  >  �  R  �  �  �  �    
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  `  A    �  �  �  v  G    �  �  N    `  {  {  u  m  e  W  C  *  
  �  �  N  �  �    }    �  w  �  �  �  �  �  �  �  �  �  Z  ,  �  �  y  .    �  {  _    �  �  �  �  �  �  �  �  {  m  i  o  u  {  �  }  w  q  k  e  h  X  H  6  !    �  �  �  �  �  �  ~  u  f  P  :    �  �  `  �  �  �  �  �  �  �  N    �  ~    �  [  �  �  �  j  i    �  �  �  �  �  y  c  N  7      �  �  �  �  �  �  V     /  A  �  �  �  �  �  �  �  v  7  �  8  o  
w  	h  7  �  |  �  �  �  �  �  �  �    x  k  T  <       �  �  �  I     �  N  t  m  g  a  [  S  K  C  :  1  (        �  �  �  �  �  `  )  �  �  �  \  3    �     *  '    �  �  �  �  C  �  Q  �  �  �  �  �  �  �  �    t  m  f  _  Z  Q  J  F  C  =  9  9  N  �  �  ^    �  \  �  �  #  
�  
(  	�  	    �  ;  �  �  �