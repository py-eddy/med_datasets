CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ǯz�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �}�   max       =�h      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E���R     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v�33334     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @O�           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �q��   max       >C��      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��a   max       B,k�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{(   max       B,Q�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?W�%   max       C��       �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P��   max       C���      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�I�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?�$�/�      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �}�   max       =�h      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�=p��
     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @v�33334     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @O�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?�:��T     �  K�                  
         �      "      !                  !   �            .   O   a      /   B      O            $         
      )      	   W      
      -   
      ,   N�ƼO%O���OY.N^�N���NV�O��P"�O+JzO]B�NޤObq�P.�KO�R<Nި�N�TO��[O��*PmO�N��O��4O�S�O��P���P���N'��O��O��LM���P"�N#/�NZ�8N�L�P�YNi8
OpXYN��qNg�Ps*N�0�N��@PNӻN_@<N���O
^iO��FN��:O|jOUz_N=J�}���ͼ49X���
���
%   ;�o;�o;�`B;�`B<o<o<t�<D��<���<�9X<�9X<ě�<ě�<ě�<�/<�`B<�`B<�`B<�h<�h<��=C�=C�=C�=\)=t�=t�=��=�w=#�
=H�9=T��=T��=aG�=e`B=ix�=u=y�#=�t�=��-=��w=�9X=���=�G�=�h��������������������|xt~��������������|����������������������������������������BABFO[a[XODBBBBBBBBB"/;<;;1/"�������������������������(������(-5B[g��������t[NB-(WYagcnz���������znaW�����������������������������������������������������������������)6HMV[WO6�����)5NZUWSNB5)�������zpqz�����������������������������uy������������������������
		��������������	��������_bhjorrtvw������tph_��������

�����#)<KPQMG</#
"#+<HYagig_[UHB<63%"�����41(����������!4<=82������������	�������
������"'.6BOTZ]_[VOLB6)��������������������_[glz������������tg_xyz��������zxxxxxxxx��������������������><BBOS[cgd[OCB>>>>>>�����5?HKJ@5)���������� ������������������������������FABHUaada]UHFFFFFFFF[S[hpt���wutkhe[[[[[������)5?A@<5.���������������������������������������������������������������������������� 
"#)+(#
������������ �������~|�����������������~mha_^\]_ajmnorttnmmm
)6BC;86-)�������������������������������������
���
�����������������������ع�����'�-�'�%�����������������������������ŭŠŇŅŉŔŠŹ���Ҿ�����������s�f�`�Z�M�A�@�H�M�Z�d�f�j��l�y�������������y�n�l�i�l�l�l�l�l�l�l�l�T�a�j�m�t�x�w�m�c�a�T�R�K�O�T�T�T�T�T�T������	�	�	������������������������������������ݼмʼü������������Ƽʼּ��B�[�l�{Ĉą�y�j�[�)��������������)�B���������������������������������������������ûػܻ���������ܻлû���������������������������������(�5�A�G�N�R�S�N�L�A�5�(��	������(���"�;�T�j�W�*� �	����׾ʾ��̾оϾ�����Z�f�s�w�����������s�f�Z�V�K�F�I�J�O�ZE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�F E�E�E����������������������������������������Ҿ����ʾ׾ܾ��߾׾ʾ�������������������������������������������f�\�V�[�f�s����ɺ����3�?�9�-���ɺ������}���������ɼ'�4�=�@�M�Y�f�o�f�Y�Q�M�A�@�4�-�(�'�%�'��/�;�H�T�]�b�T�E�/�"��	�����������	��������������������������������~�����������������������ùìåäìðù�������a�z���������r�H�"�	���������������	�/�aňŔŐ�{�b�0�
����������������0�<�b�tň������������üùùù�������������������߻����������������������x�l�`�e�t����������ûл����������ܻлû�����������ÇÓàéàØÓÇÆÅÇÇÇÇÇÇÇÇÇÇ���������ѿ���Ύ���y�m�f�i�v�{�}�����s�������������s�g�e�g�o�s�s�s�s�s�s�s�s�N�Z�g�j�g�_�Z�N�J�A�7�@�A�M�N�N�N�N�N�N�z���������������z�m�e�f�m�n�z�z�z�z�z�z�����ѿٿѿʿ������������m�[�T�R�W�`�y���5�B�N�[�\�[�O�N�B�5�3�2�5�5�5�5�5�5�5�5���Ľнݽ���������߽нĽ������������B�N�[�f�d�[�Y�N�H�B�:�=�B�B�B�B�B�B�B�B�-�:�?�A�F�O�G�F�F�:�-�+�!��!�-�-�-�-�-����$�0�;�0�#����������ưƧƢƥƱ���彅���������������������z�y�r�y�z���������<�F�H�U�Y�a�d�d�a�U�H�<�/�,�#� �#�/�2�<�U�a�nÓù��ýÞÓ�z�U�H�/����#�I�R�UFJFNFVFcFdFcFVFPFJFEF=F1F'F1F=F=FJFJFJFJǡǭǲǳǭǭǡǔǈǂǈǍǔǘǡǡǡǡǡǡDoD{D�D�D�D�D�D�D�D�D�D�D�D{DwDoDiDbDgDo��'�4�M�f�r�������f�Y�M�@�6�'���������������ûлܻ�ܻл̻û��������������~���������������������~�{�r�o�l�d�e�s�~�B�N�[�g�t�w�x�w�t�n�g�[�N�E�B�5�4�5�:�BE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� g   W R [ C I ~ 2 < P E  U = m > , ! ) ` ;   5 9 C b 5 ? L ? P ~ , 8 1 &  ` I ? F < M L > ? ~ \  b    �  b  \  �  [  �  @  �      �  $  �  P  n  #  �      �  �  �  X  {  �  �  s  3      �  Q  }  �  z  w  �  �  �  �  �      v  �  8      T  �  ��q��%   <���<�t�:�o<t�;ě�<���>C��<��
='�<T��=,1=,1=�P=��=��=#�
=e`B>A�7=�P=<j=q��=��=�/>   =8Q�=��T=���=#�
=�x�=��=�w=ix�=��P=@�=��
=y�#=ix�=ě�=�O�=�+>�+=�t�=��=�"�=��#=ȴ9=�`B>�=��#B�	Bl�B�B"�B�A��aB�PB�>B	�BZ�B 6/B"A�B�xB�wB��BaB�B��B�XB"�]BAHBՁB�B��B�OB��B�B.�B0vB"�B8�B @ZB��B�7BEB��B @B�B�B�B,k�B��B7iBȨBIRBf�B<B�dB��B��B�LB*}B�FB��B?�B��A�{(BQB�B��BI)B A4B"B�B�BB�B�qB=�B 8B�VB��B"��B��B�B7�B��B��B"B1B6rB?�B"G�B<�B �BZ!B�B��B��B�)B�QB��B@HB,Q�B��B�QB� B��B��B|B�dB�]BB8�A�4?W�%A��AB�IA)�A�zhA��A�CA�ϭA��@���@��sA���A\$�AA��C�s�A�A�AO8CAE�n@E�@�C�A���A��]A���A���A�TA��@�.G@�&�AʐAtA��`A� 4A��(Ap�A��A)Q]A�� @{r\BJ�A�uA�&�AǊ�C�� B��C���@Ԁ�@��@��A�$C�Q'A�~?P��A��AB��A3�A��A�xzA��A�X%A�{E@��S@���A���A^��AAY+C�h�A�uQAP�CAE�@L�@яnA���A�v�A���A�i�A�%Aε�@�#J@�-AʜAt��A���A��}A��sAo��A��A(��A���@xU�B�JA��A�4A�j:C���BI>C��@�٩@�V;@�mA��C�G                  
         �      "      "                  !   �             .   P   a      0   C      P            $          
      *      
   X            -         ,            '                  5               -                  5      #         C   ;               '            '               %         1            %                     #                  %               -                  #               7   5                           #               #         %            %            N�ƼN�_�O�.rN6�9N^�N���NV�Nx��PM�Nɿ�N<LNޤO7NzP&|�Or<Nި�NS�%Oh^O2
�O�MN��Oo�:On��N��Ph��P�I�N'��Ov3�O9�vM���O��\N#/�NZ�8N�L�O��kNi8
N�=N��qNg�O��7N���N��@P�cN_@<N���O
^iO��FN��:O|jOUz_N=J  �  d  �  G  �  ?  8  �  �  R  �  #  �  �  �  �  �  �  �  I  �      �  �  
3  ;  �  	�  `  
�  �  �  b    �  A  �    �  �    }  �  !  2  g  �  &  
�  8�}󶼛��o<o���
:�o;�o<#�
=��<49X<�/<o<T��<T��<�9X<�9X<�`B<���=o=�v�<�/=+=C�=P�`=H�9=C�<��=��=<j=C�=ix�=t�=t�=��='�=#�
=y�#=T��=T��=ix�=m�h=ix�=��
=y�#=�t�=��-=��w=�9X=���=�G�=�h���������������������|{���������������������������������������������������������BABFO[a[XODBBBBBBBBB"/;9/."��������������������������������������<67;BN[gt~�����tg[N<fdknpz���������ztnff�����������������������������������������������������������������)6GKTYUO6���)5KOOLB<5)��������zpqz�����������������������������wz�����������������w������������������������� ��������_bhjorrtvw������tph_������

������	
#/<HKMIB</#
	+,/5<EHTUUUOH<3/++++������()��������������
*2;5,��������	������������%&(),26BOTXZVQOB60)%��������������������ssv{��������������vsxyz��������zxxxxxxxx��������������������><BBOS[cgd[OCB>>>>>>����5>FIH?5)����������� ������������������������������FABHUaada]UHFFFFFFFF[S[hpt���wutkhe[[[[[����)5=@?;5+������������������������������������������������������������������������������ 
"#)+(#
������������ �������~|�����������������~mha_^\]_ajmnorttnmmm
)6BC;86-)�������������������������������������
���
�����������������������ع������!� ���������������������������������ŭŔŉōŔŠŹ������s�������w�s�h�f�]�f�k�s�s�s�s�s�s�s�s�l�y�������������y�n�l�i�l�l�l�l�l�l�l�l�T�a�i�m�r�v�m�a�U�T�M�P�T�T�T�T�T�T�T�T������	�	�	�����������������������������������ּʼƼʼ˼ּ޼�������)�B�O�[�e�n�o�n�g�[�O�B�)�����������)�������������������������������������������ûλлջлλû����������������������������������������������(�5�A�D�N�P�P�N�G�A�5�(����������"�;�T�g�`�T�)��	����׾ʾþ;Ҿо���f�s�t���������s�f�Z�R�M�J�M�M�R�Z�`�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�F E�E�E����������������������������������������Ҿ��ʾ׾ھ��޾׾ʾ����������������������s���������������������s�f�d�]�c�f�o�s�������#�'�"�����ɺ�������������'�4�=�@�M�Y�f�o�f�Y�Q�M�A�@�4�-�(�'�%�'�/�;�H�S�S�L�H�<�/�"��	���������	��/������������������������������������������������	�����������������������������a�u�}�u�y�}�p�a�H�"�	���������	��/�H�a��#�<�bŇőŎ�{�b�I�0�����������������������������üùùù�������������������߻��������������������������x�l�c�h�w�����ûлܻ�����������ܻлû�����������ÇÓàéàØÓÇÆÅÇÇÇÇÇÇÇÇÇÇ�������Ŀ̿ٿܿͿ��������y�w�u�y���������s�������������s�g�e�g�o�s�s�s�s�s�s�s�s�N�Z�g�j�g�_�Z�N�J�A�7�@�A�M�N�N�N�N�N�N�z���������������z�m�e�f�m�n�z�z�z�z�z�z�����Ŀпƿ������������m�]�U�T�Y�`�m�����5�B�N�[�\�[�O�N�B�5�3�2�5�5�5�5�5�5�5�5�Ľнݽ޽������ݽнŽĽ����������Ľ��B�N�[�f�d�[�Y�N�H�B�:�=�B�B�B�B�B�B�B�B�-�:�?�A�F�O�G�F�F�:�-�+�!��!�-�-�-�-�-�����$�)�$�!����������ƴƧƣƦƳ���彅���������������������{�y�t�y�}���������<�F�H�U�Y�a�d�d�a�U�H�<�/�,�#� �#�/�2�<�zÓìóó×ÓÄ�z�a�U�H�;�/�&�.�<�U�a�zFJFNFVFcFdFcFVFPFJFEF=F1F'F1F=F=FJFJFJFJǡǭǲǳǭǭǡǔǈǂǈǍǔǘǡǡǡǡǡǡDoD{D�D�D�D�D�D�D�D�D�D�D�D{DwDoDiDbDgDo��'�4�M�f�r�������f�Y�M�@�6�'���������������ûлܻ�ܻл̻û��������������~���������������������~�{�r�o�l�d�e�s�~�B�N�[�g�t�w�x�w�t�n�g�[�N�E�B�5�4�5�:�BE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� g  [ : [ H I R 5 9 0 E  U ; m ? -   ? ` +   D B b 4 8 L 3 P ~ , 0 1   ` 9 @ F A M L > ? ~ \  b    �  �    Z  [  �  @  �  h  �  O  $  x  E  �  #  u  �  t  �  �  �  �  �  �  m  s  �  �    T  Q  }  �  ;  w    �  �  F  �    h  v  �  8      T  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  l  [  �    M  Z  b  b  X  H  .    �  �  �  _  '  �  �  o  X  P  �  �  �  �  �  �  �  �  c  B  /  �  �  �  f  (  �  s  -  �  �            !  '  0  7  <  D  '  �  �  :  �  �  6  �  �  �                    �  �  �  �  �  �  �  �  �  1  8  >  9  3  *  "    	  �  �  �  �  q  A    �  �  4   �  8  :  ;  =  ?  @  B  C  E  F  I  L  O  R  U  X  [  ^  a  d  {  �  �  �  �  �  �  �  �  �  s  I    �  �  F  �  �  �  X  �  h    �    h  �  �  �  t  $  �  (  F    �  
�  	2  �      %  4  @  H  N  R  P  =  $    �  �  �  s  8  �  l   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  1  �  m    �  #  &  )  ,  ,  "        �  �  �  �  �  �  q  T  6    �  �  �  �  �  �  �  �  �  x  \  ;    �  �  R  �  �    j  A  �  �  �  w  h  [  N  >  *    �  �  �  �  X  N  "  �  o  D  �  �  �  �  �  �  �  �  |  \  =        �  �  �  �  C  �  �  �  �  �  �  �  q  V  :  -  (      �  �  �  �  J    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Q  ,  �  �  �  �  �  �  �  �  }  j  Z  H  4    �  �  �  p  B    �  �  �  �  �  �  �  �  �  �  �  �  p  N  &  �  �  `    �  w  Q  	�  
�  l    x  �    =  I  4  �  �  d  I  �  W  
E  �  _  �  �  �  �  �  �  �  �  �  u  Y  >  $    �  �  �  �  �  �  �       �  �  �      �  �  �  �  �  �  o  K  +    �  �  �  �  �        	  �  �  �  �  o  E    �  �  o    �  !  �  J  H  S  |  �      �  �  �  �  �  �  �  O  �    ?  V  a  �    1  _  �  �  �  @  �  5    �  ~  9  �  p    �  �  �  
  
3  
  	�  	�  	�  	:  �  �  :  �  C      �  w  �  
    �  ;    �  �  �  �  �  �  x  ]  B  *       �  �  �  �  �  s  �  �  �  �  �  �  �  Q    �  �  \    �  0  �  &  �     i  	i  	�  	�  	�  	�  	�  	�  	�  	�  	�  	m  	7  �  �  �       �  �  o  `  R  C  5  "    �  �  �  �  �  �  �  �  �  x  n  f  ^  V  	�  
B  
u  
�  
�  
�  
�  
�  
�  
�  
I  	�  	�  	9  �        �  �  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  \  I  7  $    �  �  �  �  �  t  Y  b  b  ]  W  Q  G  ;  -    
  �  �  d    �  C  �  �    �        �  �  �  �  �  r  H    �  y    �  p    �  8  �  �  �  �  �  �  �  �  s  e  V  F  7  &      �  �  �  �  �  �  �      *  3  ;  A  @  9  &    �  �  [    �  �  ;  �  �  �  �  �  w  l  d  \  T  I  ;  ,      �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  S  �  �  {  d  F    �  �  �  M    �  d    �  ?  �  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  u  S    �  �  A  �  S      �  �  �  �  �  l  N  (  �  �  �  |  j  h  �  �  �  �  
G  
�  4  h  {  x  _  5  
�  
�  
<  	�  	  v  �  W  �  �  5  �  �  �  �  �  �  �  s  P  *    �  �  ~  L    �  �  Z     �  !     �  �  �  m  E  (  
  �  �  �  �  a  :    �  �  �  ~  2    �  �  �  �  e  ;    �  �  \    �  [  �  �  a  �  =  g  _  O  >  6  -      &    �  �  �  _  !  �  j     P  �  �  �  �  w  K  +    �  �  �  �  E  t  k  `  U  E  3    �  &  
  �  �  �  y  O    �  �  z  D    �  i    �  �  =   �  
�  
�  
x  
O  
!  	�  	�  	  	F  	  �  �  ;  �  ]  �    \  �  /  8    �  �  �  �  �  h  I  *    �  �  �  �  �  �  �  �  �