CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�fffffg      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�E�   max       P�a$      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =��      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @F�\(�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ᙙ���    max       @vep��
>     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O�           d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�{�          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >o��      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��B   max       B4��      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�,�   max       B4�      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?l9�   max       C��      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?m�D   max       C�(      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�E�   max       P	�      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��kP��|   max       ?���Q�      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       =�S�      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @F�\(�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ᙙ���    max       @vep��
>     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�3           �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CO   max         CO      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?��ᰉ�(     P  J                  	   -      Q      P   -                        Y               Y                                                <      Q   �         $      -N&OO ��N#��O��N��N��XO���Nw�AP]�OsYlP�a$PU �N���N:T�O��O�f@N�5O�&&NDO���Oi>�On�	P+Nz�P5�EN�0�Ob�N��O8D�N��#N>�}O���M�E�N�XNꅤNV�N)$O6�N��Oj?�P��N@DP�LPt��OY��N�	OE��O ��O��ҽt����
�u�49X�D����o��o;o;�o;�`B<#�
<#�
<e`B<u<�C�<�t�<�1<�9X<ě�<ě�<���<�/<�`B<�`B<�h=+=C�=C�=C�=C�=\)=t�=�P=��=��=�w=#�
=,1=<j=D��=D��=]/=]/=q��=�o=��=���=��w=��~�����������~~~~~~~~vz����������������vv))6BIMB6/)))))))))))ACCBBA@BORV[`dhf[OBA`hhrt���������th````��������������������������
#<@JMJ#
���PPX[gltvtqg[PPPPPPPPOJFKQ[ht��������th[O�������������������� �)BNgt�����NB<" ��
#<HalujUF8/	�� #./<CCCD></#��������������������((&/4<>HSU`ab[UH<1/(CDHO[]t��{vv}�th[TOC��������������������4,35B[ccbfgztg[HB>4"##)5865/)""""""""""�������
	���������)35BD@5)�dgt�������������ytgd��������������!)+/))$����������������������������������������()*05BFNSWONLB54-))(��������������������)6;;9:861)��������������������xuqqtz{{�����{xxxxxx���
#/<BLH<#��������������������������������������
#+-*#
�����������������������fgtt����tsigffffffff������������������������������� ����������������������������������	��������&)+-*))6[t��xh[OB53/+%{�������� �����{������������������������


�������``b``afnz�������znb`\VVZabmz������zxma\\�������	 �����0�7�<�@�=�<�0�,�#�!�#�*�0�0�0�0�0�0�0�0ĦĳķĿ����ĿľĳĪĦĢĚĕĔĕĚĞĦĦÇÉÐÎÇ�z�v�q�zÆÇÇÇÇÇÇÇÇÇÇ�y�����������������������������y�p�m�u�y�L�W�Y�e�h�p�p�n�e�Y�S�L�I�D�D�H�L�L�L�L������������������������������������ٿ;�G�T�`�y�����������y�`�G�;���'�(�.�;�����������������������������������������s��������̾ؾݾݾ׾������f�Z�E�9�9�H�s�ݽ�����������ݽн�����������������;�L�J�N�[�^�H�A�;�/��������������	������Ǿ;Ǿ��������s�a�M�4��"�4�s�����D�D�EEEEEE*EEED�D�D�D�D�D�D�D�D��������׾̾Ҿ׾�����������������������������������������뻑�������������Ż��������x�F�<�A�W�x������������������������������������������ҿG�`�y�������y�m�`�T�G�;�.�-�-�)�.�3�;�G��*�6�B�<�6�*�(������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��O�\�h�u�yƁƉƎƕƎƁ�}�o�h�\�P�G�G�L�O����������������s�f�K�?�>�M�Z�f��������6�N�S�J�C�����׿ҿԿοѿܿ����(�6�����ÿĿɿĿ���������������������������ù�������������ùÇ�z�n�a�^�a�n�vàù�/�<�H�N�T�H�E�<�6�/�-�#�� �#�,�/�/�/�/����$�+�)�$�����������������������(�4�A�M�Z�V�N�M�A�4�-�(��������e�r�~�������������������~�r�k�e�`�_�Z�e��������������ܻۻܻݻ����齫���Ľнݽ���ݽнĽ������������������N�[�g�t�~�~�t�q�l�k�g�N�?�5�!�!�)�5�B�N�"�.�;�;�;�3�.�"���"�"�"�"�"�"�"�"�"�"�����������������������(�+�(�(�(�*�(�&����������
��ÇÓàìñìãàÓÇÆÆÇÇÇÇÇÇÇÇ�B�D�B�B�:�6�)�&�)�)�6�;�B�B�B�B�B�B�B�B����'�/�3�6�3�'�������������������ʼּڼ�ּӼʼ����������������������(�5�A�H�N�U�^�d�f�Z�N�A�5�(������(��������*�.�,�$��������������t�l�������n�s�{ŇŉŔřŔŇ�{�y�n�j�b�n�n�n�n�n�n�f������~�f�M�@�4���������4�@�f�������ʼ����� ����ʼ��������s�p�p�t������������������������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�EyEuEsEuEuEuEu���������ɺκֺݺݺ׺ɺ�����������������ŔŠŭŹ������������ŹŭŠŜŔŔœőŔŔ²¿������������������¿²¥² @ < P h ` \  , : n > B 8 t 5 n 0 N W # + P E 3 A 6 - E E 9  ( A n S B n  T C W z R : V ; 0 ! (    M  7  ^  /  A  �  .  �  �  c  <  �    t  Q  �  �  ?  D  �  �    �    K  �  +    �  �  �  A    j    b  {  }    �  4  �  �  �    �  �  V  �o:�o�49X;o<u<o=0 �<t�=�1<�`B=�Q�=aG�=#�
<���=,1=8Q�=+=��<�/=�`B=�P=�P=Y�=+==H�9=,1=e`B=@�=aG�=#�
=�+=�w=,1=H�9=]/=0 �=��=u=��w=�"�=u>	7L>o��=�Q�=�1=�S�=�v�>�PB
�B�B�B��B��B�JB��B	�B;B!@Bt�B;�BH�B4��B��B��B�"BhbB��B�Ba�B
��B�BLcB�bB�eB�=B�B��B �B(�AB��B��B�#B�B"t�B	�B!P�B#@�B4&B,�B4�B��B��B�EB�B,�A��BB�VB
��B?�B�B��B�$B@ BBSB��B?ZB!?�B@qBD"B?�B4�B�lB�!B��B��B��B��B�B>iB��B=�B��B§BC=B?�B?�B �HB(�BG�BńB3�B@�B"@B	�sB!@B#@yB>%B>�BCZB��B��B=bB?�BA6A�,�B��A��A���A��A��?�r�B��Ah�;A�1AF�RA(w�A�/XAE��C�U�AU7�A�Cq@��A��OAgN�A��C��
B*{AB�eA���Av3A���A�3B��A9�@?�@�T�A'��A�qA`XhA��nA��'A��A׿_?l9�@��*A��A�1A��@�>@��fA�+�C��@(�1A��A�l�A�EA���A�@QA_?���B��Ai2�A�y�AF�JA+9A�{#AE�C�K�AT��A�U�@��AМ�Ah��A���C���B��AB�#A��Av�A��2A�[0B�rA:�e@�~@��-A'�A��A`A�zoA��ZA�VmAׁ,?m�D@��HA��OA�V�A�}"@˵�@��{A�QC�(@, �A��A��2                  
   -      Q      Q   .                        Y               Y      	                                          <      Q   �         %      .                     %      )      3   3            #                     )      -                                                -      -   3                                                   )            #                     '                                                      '      )   %               N&ON�McN#��N Nx0N��XO��Nw�AOq	
O`�O�2�P	�N��'N:T�N���O�f@Na�sOzݽNDN�cO#�On�	P
�Nz�O�rBN��Ob�N��QO8D�N��#N>�}OXVM�E�N�XNꅤNV�N)$O6�N��OT��O�� N@DO�0�P	�O �3N�	O0�OO ��Od��  I  =  g  �  �  �  �  �    �  �  �    �      �  �  -  L    8  �  c  	�  .    A  �  �  �  �  �  a  �  �  �  R  w  &       	  �  �  
  �    ��t���o�u��o;ě���o<49X;o=#�
<o=@�<�1<u<u<���<�t�<ě�<�j<ě�=y�#<�h<�/<�h<�`B=�\)=C�=C�=\)=C�=C�=\)=,1=�P=��=��=�w=#�
=,1=<j=L��=e`B=]/=y�#=�S�=�hs=��=��-=��w=�;d~�����������~~~~~~~~~���������������~~~~))6BIMB6/)))))))))))HIKO[]a\[OHHHHHHHHHHtot}�������ttttttttt������������������������
"/7<BCB9/#�PPX[gltvtqg[PPPPPPPPYWX\cht��������tha\Y�������������������� !+5BO[pt}���o[NB9. ��#<addfaU</#
� #//<BBBC=</#��������������������/-/2<HLUWUSH<<</////CDHO[]t��{vv}�th[TOC��������������������51-55B[abafg{tg[KB?5"##)5865/)""""""""""������

���������	)/5@=5,)	dgt�������������ytgd��������
������!)+/))$����������������������������������������()*05BFNSWONLB54-))(��������������������)6;;9:861)��������������������xuqqtz{{�����{xxxxxx��
#0:<A@3/#
������������������������������������
#+-*#
�����������������������fgtt����tsigffffffff������������������������������� �������������������������������������������&)+-*)")BVdp~�thb[OB861-���������	�����������������������������������


�������a`agnz��������znddaa\VVZabmz������zxma\\�����������0�7�<�@�=�<�0�,�#�!�#�*�0�0�0�0�0�0�0�0ĦĳĳĿ����ĿĺĳĦĚęĖĘĚĥĦĦĦĦÇÉÐÎÇ�z�v�q�zÆÇÇÇÇÇÇÇÇÇÇ�y�������������y�x�s�y�y�y�y�y�y�y�y�y�y�L�Y�`�e�h�e�d�Y�V�L�K�L�L�L�L�L�L�L�L�L������������������������������������ٿ`�m�y������������y�p�`�T�G�;�3�/�7�G�`���������������������������������������������������������������s�f�`�V�Z�f�s��ݽ�����
������ݽн�����������������"�/�5�9�;�8�/�,�"��	����������������������þ¾���������w�l�G�9�4�A�f�s����D�D�EEEEEE(EEED�D�D�D�D�D�D�D�D��������׾̾Ҿ׾������������������
���������������������������뻑�������������Ż��������x�F�<�A�W�x�������������������������������������������ҿG�T�`�y�{���~�y�m�`�G�E�;�0�.�*�.�5�;�G��*�6�B�<�6�*�(������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��\�`�h�u�}ƁƆƉƁ�u�h�\�S�O�M�J�M�O�V�\����������������s�f�K�?�>�M�Z�f���������5�O�Q�P�H�A�5�����߿տֿпҿ��������ÿĿɿĿ���������������������������ù��������������������ùìÛÓÍÏØìù�/�<�H�L�S�H�D�<�4�/�-�#� �!�#�-�/�/�/�/����$�+�)�$����������������������(�4�A�M�Y�V�N�M�A�4�/�(�����(�(�(�(�e�r�~�������������������~�r�k�e�`�_�Z�e��������������ܻۻܻݻ����齫���Ľнݽ���ݽнĽ������������������B�N�[�g�p�t�t�m�g�[�V�N�B�5�2�'�&�)�2�B�"�.�;�;�;�3�.�"���"�"�"�"�"�"�"�"�"�"�����������������������(�+�(�(�(�*�(�&����������
��ÇÓàìñìãàÓÇÆÆÇÇÇÇÇÇÇÇ�B�D�B�B�:�6�)�&�)�)�6�;�B�B�B�B�B�B�B�B����'�/�3�6�3�'�������������������ʼּڼ�ּӼʼ����������������������5�A�F�N�T�\�b�c�Z�N�A�5�(������(�5�������	��%�)�'��	���������������������n�s�{ŇŉŔřŔŇ�{�y�n�j�b�n�n�n�n�n�n�f�r�{�r�f�Y�M�@�4�����������4�f�����ʼ�������ּ�����������������������������������������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�EyEuEsEuEuEuEu���ɺֺֺֺ̺ܺܺɺ���������������������ŔŠŭŹ������������ŹŭŠŜŔŔœőŔŔ²¿����������������¿²¯¦ ¥² @ 5 P 3 M \ . , . l 7 > 6 t ' n * J W  + P C 3 2 8 - > E 9   A n S B n  T C C z L 2 3 ;  ! !    M  �  ^  :  >  �  :  �  �  G  �  �    t  �  �  t  
  D  �  `    �      �  +  �  �  �  �  �    j    b  {  }    �  B  �  F  i  )  �  r  V  �  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  CO  I  G  D  B  ?  <  9  5  1  -  +  *  )  (  '  &  &  %  %  $  %  0  7  <  ;  1      �  �  �  d  0  �  �  z  .  �  �  ,  g  _  W  P  H  @  8  0  (           �  �  �  �  �  �  �  [  a  e  i  �  �  �  �  �  �  �  �  �  �  �  v  a  K  5    �    A  \  u  �  �  �  �  �  �  �  z  E    �  �  B  �  �  �  �  �  u  d  O  :  %    �  �  �  �  �  d  ?    �    !  O  N  H  D  �  �  �  {  X  +  �  �  Y    �  �  �  �    @  �  �  �  �  �  �  �  �  �  �  v  c  O  <  )      �  �  �  5  �    _  �  �  �  �      �  �  d    �    �  �  i  V  �  �  �  �  �  �  �  �  �  �  z  _  5    �  }  "  �  *   �  �  �  a  �  �  4  p  �  �  �  �  �  ]    �    j  �  t  �  @  d  x  �  �  �  �  u  L    �  �  �  F  �  �  F  �  u   �                  �  �  �  H  �  �  A  �  Z  �  _  �  �  |  v  p  i  c  ]  W  Q  K  B  5  (        �  �  �  �  >  m  �  �  �  �      �  �  �  �  j  *  �  �  @  �  �  �    �  �  �  t  �  �  �  �  �  �  �  �  ;  �  b  �  3  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  V  <  '      �  �  �  �  �  �  �  �  �  }  f  I  )    �  �  �  p  9  �  -  )  %  !            �  �  �  �  �  �  �  a  ;     �  �  f  �  E  �  �  &  D  L  <    �  V  �  
    	�  [  �  �  �  �                   �  �  �  �  �  f  C    :  p  8  *             +  ,         �  �  �  �  �  q  ^  M  �  �  �  �  �  �  �  �  �  �  |  f  C    �  �  x  6  �    c  Z  P  G  <  /  "      �  �  �  �  �  �  �  j  T  ?  )  W  �  	/  	g  	�  	�  	�  	�  	�  	�  	�  	�  	W  �    �  `  R  i  /  -  -  -  +  !    �  �  �  �  o  F    �  �  �  L    �  �              �  �  �  �  �  �  �  �  j  H  &     �   �     ?  /      �  �  �  r  J  !  �  �  �  �  }  U  *  	  �  �  �  �  �  �  w  e  O  M  H    �  �  �  Y  2     �  �  T  �  �  �  �  �  �  �  �  {  d  G  *    �  �  �  ~    O  W  �  �  w  `  H  /    �  �  �  �  �  p  Y  A  )    	   �   �  �  �  �  �  �  �  �  �  �  �  �  �  |  >  �  �  -  �     e  �  �  �  �  �  �  �  �  �  �  �  u  g  Z  M  @  2  %      a  d  h  l  p  u  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  Z  K  <  -      �  �  �  �  �  �  b  =    �  �  �  �  �  �  �  u  X  9    �  �  �  _  -  �  �  �  ^    �  �  �  �  �  �  �  �  �  �  �  �  x  j  [  G  3       �  �  R  ?    �  �  �  �  �  n  I     �  �  u  1  �  �  �  :  �  w  q  n  k  c  e  b  [  S  K  D  =  5  1  H    �  �  �  �  "  &  %         �  �  �  �  Z  $  �  �  W    �    `  h  �        �  �  �  �  E    �  �  c  5  
  �  y  �  �  ,      �  �  �  �  �  �  �  �  ~  Q  %  �  �  �  y  L    �  �  �  	  �  �  �  �  �  	  	
  		  	  �  �  �  L  �  �      �  �    �  �  �  �  �  �    �  o  J    T  p  W  	�  	  J  �  �  �  �  �  �  �  �  �  �  �  u  G    �  �  9  �  4    
    �  �  �  �  x  T  .    �  k    �  M  �  T  �  H   �  �  �  �  �  �  �  [  )  �  �  y  ?     �  q    {  �  
  =    �  �  �  �  n  D    �  �  �  K    �  �  R    �  B    a  �  �  �  �  �  �  q  N  !  �  �  O  �  t  �  1  Q  )  