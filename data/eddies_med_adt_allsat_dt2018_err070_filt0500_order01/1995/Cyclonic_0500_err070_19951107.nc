CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @F��\)     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v9��R     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3�        max       @P`           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �bN   max       ��o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�T   max       B4��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B4�4       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >5\   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@x    max       C��.       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          W       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �   max       Pk�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u�!�R�   max       ?̎�q�i�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <��
       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @F��\)     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v9��R     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�I�           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Aj   max         Aj       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?̎�q�i�     �  ]   '               
            W   	                "   &      +                           /      
   3   8   '   /                  ;   	            &                                       (   G            !   
                  O�X}O�YO�O?�N&�5NJ�LN��SN���N�,8P���O�N�QHNl�KO�jO�9�P�O�߷NS��O�ІN���N �N.�O>�N��N���O;�;O@�P8]O��N��KP8�oO�gvP?�'O{��N���ON��OjO��gO2��PR�N�>7N�qOO��cN8��P��N���O]�OX�N���OWZ@O zNʅ�Ngs�N,�4N�*�N�iN�eP$��Pqj�N&�@O��O_c$O��N�\BN���OA1NUaO`	�N���N��<���<T��;ě�;ě�;ě�;�o;o�o���
���
���
�o�t��u��o��C���C���t����㼣�
���
��1��9X��9X��j������/��/��`B��h�������+�C��\)�t��,1�,1�49X�8Q�<j�<j�<j�@��@��D���Y��Y��e`B�m�h�m�h�m�h�y�#��o��o�����+��\)�������w���
���T��1��E���E���^5�Ƨ������[am����������zrj`[[PUWZZaez�������znaUP�����������������������������������)-63)��������������������pz������������zxppppnt���������tmnnnnnn��������������������������#*;6#��������%,6CJMOQOC0*)*5BNNVV[\[SNB5,)#-.%#��������������������	
#/8?DLG</#
		Sm{}v������������z^Sz���������������~rrz����������������������������������������agt�����tkg`aaaaaaaa��

��������������������������������#)5BIN[ggb[NKB54)"!#��������������������OOQY[htyttnhb[OMOOOO#/29893/#
	jn{������������{nhfj������%!���������JO[ht�������th[OIFGJ	���������������������������"-/.)#	�������|������������������|��������������������GHTTamsyumaTQHGGGGGG�������)������������������������������������������������������������������)49;9BMN=)����nt�������������wtlnnYaimmvz{~�zmia]XYYYY�������������������������

����������&5Ngn\^]SB%
	stx{������������{tss=ILbhn{����{nbUI<3=V[ggt�������tg[YQVV��������������������QUanz�����vrnaYUTRQQ��������� ����������������������������������������������������������������������BIU\bhhb_YUMIFC=BBBBlnp{|����������{{onlLOT[honmlhih^[WSOMKL`nz������������ljma`>Ng������������zgN=>#.*# LOPTW[hlptyutjh[[OML��������������������HUaz��������vnUHDCDH�����/0<CILMLII<90/*)////07<<HILNRTTRKI<50/-0#/<<@></,#��������������������������������������������������������������������������6�C�O�C�>�*��������������s�l�g�^�`�m�s����������������������������������������#�%�9�/�*�#��
��¦¤¥¦²»¿����������¿²¦¦�ֺʺɺǺɺֺ����ֺֺֺֺֺֺֺֺֺ��U�N�J�O�U�X�a�d�g�b�a�\�U�U�U�U�U�U�U�U���������������������������������������ҽ����������������������������������������m�f�`�\�Y�`�m�y���������y�u�m�m�m�m�m�m�s�g�[�A�$�#�8�A�s���������������������s�.�-�!��	�������	��"�#�'�.�0�0�2�7�.�A�7�5�/�.�5�5�A�N�W�Z�]�Z�Y�W�O�N�L�A�Aìæçìù��������ùìììììììììì�l�c�j�n�m�g�k����������þžҾԾ�����l�/�,�#����#�/�<�U�a�u�y�u�n�d�U�H�<�/�U�@�#�����Ŀ��������0�I�\�g�_�f�a�b�U�2�������������(�A�O�Z�`�_�Z�N�A�2�����������������������������������������'���'�4�B�M�e�h�o�r�w����~�r�f�Y�M�'����������������������������������f�b�Y�W�Y�f�r�t�s�r�f�f�f�f�f�f�f�f�f�f�H�=�<�8�/�'�/�<�>�H�R�U�V�U�H�H�H�H�H�H�a�Z�W�\�a�k�t�z�������������������z�m�a���������������%�)�-�)�$������ݿݿѿĿÿ��ÿĿѿѿݿ������ݿݿݿ��B�6�.�.�4�6�B�O�[�h�k�k�t�x�~�t�h�[�O�B�������������������ļʼԼּ׼ּӼмʼ����f�^�c�y��������!�3�6�1�!����̼�����f�x�m�g�r�y�����������������������������x�ʾɾʾо׾�����������׾ʾʾʾʾʾʻ����_�F�:�1�9�F�O�l�����ûܻ��ݻлû�����������4�@�M�\�f�������r�Y�@����`�G�B�?�G�m�������Ŀʿ߿������ѿ����`���������Ŀѿݿ�����	�������ݿѿĿ����	�	���	��"�&�(�&�"�������ìáÔÇ�z�x�r�w�u�v�zÇÓßäìúýùì������������������������������	���������g�b�S�K�I�:�=�I�V�b�o�{ǉǑǖǕǑǈ�{�gŭŬŨŤŪŭŹ����������������������ŹŭÇ�z�n�U�I�U�a�zÇîù��������������àÇùõõíìâìóù������������������ùù��ƴƳƳƳ�������������������������������������������������$�0�2�8�5�0�$���������������������������ŲŰŹ�����������*�6�?�6�*�������Ų����ŹŭŢŠŗŗŠŭŹ���������������������w�s�^�Z�X�[�e�g�s�~�����������������������ݾ׾־׾����	�	���
�	�����׾̾˾˾Ҿ׾޾��������ܾ׾׾׾׾ʾ¾��������ʾ׾����	��
������׾��a�Z�]�g�v�������������������������z�m�a�����������������ɺֺںݺֺֺкɺ��������r�p�q�r�~�~���������������~�r�r�r�r�r�r��������������������������������������-�*�,�-�2�:�F�L�S�_�f�_�\�S�F�:�-�-�-�-�S�P�F�:�:�:�F�G�S�_�l�x�{�y�x�l�k�_�U�S�������z�}�������������������������������T�:�3�7�B�a�������������������������m�T��½¢¿������/�<�n�q�d�<�$�����DIDADCDIDVDaDbDfDbDVDIDIDIDIDIDIDIDIDIDI�ù������������������ùϹйܹ�ܹܹҹϹú���'�3�@�Y�e�~���������r�L�@�3�-�'������������ùϹ����������ҹù������Ľ������Ľнݽ������������ݽнĽ����������������ĽȽνĽ���������������������ݽݽ������(�4�6�;�4�,�(��D{DuDwD{D{D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{����ĿĶįĸĿ������������������������̽!���!�'�.�:�G�S�W�S�N�G�:�.�*�!�!�!�!�S�H�L�S�`�d�l�p�l�`�S�S�S�S�S�S�S�S�S�S : = p 3 - � m \ ? A t V . 3 / p J - 9 2 M o J , K ' D n i E U I X C K r _ V ) L H I , Z A = e 9 n 3 } H V 5 1 1 h D W F 9 � W e 6 P A ; ` >  �  {  �    9  �  &  �  �  P  �  /  n  �    /  �  U  L  �  &  �  �  �  �  �  @  4  �  �  ]  |  �  �  �  :  #  �  {    �  �  H  o  �      4    �  �    �  F  �    R    G  >  '  �  �  \  �  h  u  �  �  "�T���o�T���49X��o�ě����
�ě��49X��^5�T���u���
�,1�D���L�ͽaG���j�}��/�ě��+��w��/���ͽ8Q��P���P�ixս�w���
��{��O߽��
���T���D����7L������`�]/�Y���7L�T����{�]/�}󶽃o�m�h���P��\)��t���o��7L��t�������{����bN��9X������vɽ�xս�vɽ����F����;d��l���/B *iBtmBd�B]�B%$B>�B�B�kBW.B� B/�B~B�QB ��B��B �KB"�B4��B �\B	��B$�B �}BT�BWXB��B|VB)/�B-|�BefB^oB�HB�,B*o^Bm�A�TBS�B��BZB��B8LB
�A�olB�vB�BO�B
�B'��B	}EB�B��B�HB"s B"�B?B'�B)>mBb�B?�B
h�B�aB��B)BlVB B&9�B&w*B<.B!�B��Bb�B ?NB�{B@�BBB/9BIB(BB�nBFMB-�B0CB�QB�0B @0B�B ��BJ�B4�4B ũB	��B$;�B �3B@�Bw�B��B�B)2KB-�_B��B5�B̵B��B*�UB��A�~�B?�B٣BB�B�KB�&B
@+A���B�>B@=BI�B
�lB(�B	��B)�Bb�B&�B"�2B"B=@B&��B)@�B?�B�B	A2BDQBK�B��B=�B�B&?[B&@PB6�B�B�/BNSA�ܣA�xkA���A��@=�dA��9A��A��Ak�A�`A]��A���A�x-AI<�AĽA狫A�;RAJzP@�<vA�:j@�(�AÝ~A��3A��A{�WA���@�!�@� �@�HAU
@�M�@���As�SA}l�A���A�X:A�� BnlA��GA�ZA��B[~B��B��A���A�XqA��(AX\�AS��ATI,A��@/��@
�B�@��@�bRA��A�H0A���C�t>7֗?�>5\A,��A#=A3��C���A�i�A��Ak�A��A�u�A��A��@<��A�n�AЀA CtAk�A���A[��A�A�nAH�Aā2A�tA���AK��@٨%A�)@�`A�x�A���A�7�Az�"A؅k@��IA��@��AT�@�}�@���AkE�A~��A��|A��A�whB��A���A�{�A΀6B;�B�B�eA��A�|�A��AW AS>ASgA���@3�<@ӐB��@{�j@���A�U3A�>,A�~�C�iL>@�u?��>@x A.ǏA"�A4��C��.A��A7 A�h   '                           W   	            !   "   '      ,                           0         3   8   (   0                  ;   
            '                              	         )   H            "   
                     #                           5            !      7                                    5   !      +   '   7            !         +               )                                       -   ;         #   !                                                               !                                          5   !      +      7                     +               #                                       -   9         #                        O�� Oe$'N��N�%�N&�5NJ�LN��SNjNLN�,8O��eO�N��N66O�<�NʘfOpOoNS��O2bN���N �N.�N���N��N���O;�;O@�P39�O��N���P8�oO�1�P?�'OJ_)N���ON��O��O��gOo�O�D�N�>7N�qOO�"�N8��O� 2N���O]�N�9�N���OWZ@O zNʅ�N7�lN,�4N�*�N���N�	P$��Pk�N&�@NtXfO_c$O�K�N�\BN���N͉�NUaOK�N���N��  �  �  }  0  8  M  �  �  g  K  �  =  �  �  P  �  W  �  %  y  �  R  t  �  U  �  �  �  �  |  d  ]  �  Y  �  �  L  �  �  �    F  o  �  �  _  �  r  �  �  �  P  Q  ;    �  %  �  �  �  �  �  �  D  C  w  )  �    <��
<#�
;��
;�o;ě�;�o;o�D�����
�D�����
�t��#�
��o����`B�+��t��t����
���
��1��`B��9X��j������/��`B��`B�����,1����w�C��\)��w�,1�<j�P�`�8Q�<j�@��<j�Y��@��D���aG��Y��e`B�m�h�m�h�q���y�#��o��7L��C���+��hs������-���
��1��1��E������^5�ȴ9������`fm�����������zund^`aenz���������zna\[]a��������������������������� ��������)-63)��������������������pz������������zxpppppt��������topppppppp�������������������������

�������%,6CJMOQOC0*#)-5BKNUSNB54)#######)-$#��������������������"#(/5<<><7/#������������������{�������������������������������������������������������������agt�����tkg`aaaaaaaa��

��������������������������������))5BBIKCB53)(&))))))��������������������OOQY[htyttnhb[OMOOOO#/29893/#
	jn{������������{nhfj�����$$ ���������JO[ht�������th[OIFGJ��������������������������!$')& �������|������������������|��������������������GHTTamsyumaTQHGGGGGG�������)�������������������������������������������������������������������)364<C>7)����nt�������������wtlnnYaimmvz{~�zmia]XYYYY�������������������������

����������4B]c`WXUJB)stx{������������{tss=ILbhn{����{nbUI<3=Y[got}����}tg\[TYYYY��������������������QUanz�����vrnaYUTRQQ��������� ����������������������������������������������������������������������BIU\bhhb_YUMIFC=BBBBy{��������{utyyyyyyNOX[hhllkihd[YUONMNN`nz������������ljma`>Ng������������zgN=>#.*# V[hmsmh[SPVVVVVVVVVV��������������������FHJUanz����znaUPHEEF�����/0<CILMLII<90/*)/////004;<IIORRPIG<800//#/<<@></,#����������������������������������������������������������������������������6�:�B�5�*��������s�i�c�e�q�s���������������������������
���������������
��#�#�/�8�/�'�#��
�
²¦¦¦©²¸¿����������¿²²�ֺʺɺǺɺֺ����ֺֺֺֺֺֺֺֺֺ��U�N�J�O�U�X�a�d�g�b�a�\�U�U�U�U�U�U�U�U���������������������������������������ҽ����������������������������������������m�f�`�\�Y�`�m�y���������y�u�m�m�m�m�m�m�s�g�W�L�I�M�Z�g�s���������������������s�.�-�!��	�������	��"�#�'�.�0�0�2�7�.�A�;�5�1�/�5�7�A�N�U�U�N�N�G�A�A�A�A�A�Aìèéìù��������ùìììììììììì�n�e�k�o�n�h�n����������¾ľо˾�����n�H�<�<�/�/�/�0�<�H�I�U�X�a�j�f�a�U�Q�H�H���������������������#�0�7�0�#��
��������������(�5�6�A�K�K�A�5�(��������������������������������������������M�A�4�'�$�'�+�4�@�M�Y�f�p�r�t�r�k�f�Y�M����������������������������������f�b�Y�W�Y�f�r�t�s�r�f�f�f�f�f�f�f�f�f�f�H�=�<�8�/�'�/�<�>�H�R�U�V�U�H�H�H�H�H�H�a�a�\�a�a�m�z���������{�z�m�a�a�a�a�a�a���������������%�)�-�)�$������ݿݿѿĿÿ��ÿĿѿѿݿ������ݿݿݿ��B�6�.�.�4�6�B�O�[�h�k�k�t�x�~�t�h�[�O�B�������������������ļʼԼּ׼ּӼмʼ����_�d�z���������!�3�5�0�!����ȼ����r�_�x�m�g�r�y�����������������������������x�ʾʾʾԾ׾�����������׾ʾʾʾʾʾʻ����_�F�:�1�9�F�O�l�����ûܻ��ݻлû��� ������'�4�@�M�Y�^�y�x�r�f�M�@���`�G�B�?�G�m�������Ŀʿ߿������ѿ����`�Ŀ��������Ŀݿ�����
��������ݿѿ����	�	���	��"�&�(�&�"�������ìáÔÇ�z�x�r�w�u�v�zÇÓßäìúýùì�����������������������������������������g�b�S�K�I�:�=�I�V�b�o�{ǉǑǖǕǑǈ�{�g������ŹŭŬŧŭŮŹ��������������������ÓÇ�}�n�]�Q�a�nÇìù��������������ùÓùõõíìâìóù������������������ùù��ƴƳƳƳ�����������������������������������������������$�0�7�5�0�+�$��������������������������������ż�������������*�6�:�3�*��������������ŹŭŢŠŗŗŠŭŹ���������������������w�s�^�Z�X�[�e�g�s�~������������������������������	���	��������׾̾˾˾Ҿ׾޾��������ܾ׾׾׾׾ʾ¾��������ʾ׾����	��
������׾��a�Z�]�g�v�������������������������z�m�a�����������������ɺֺںݺֺֺкɺ��������~�t�u�~�����������������~�~�~�~�~�~�~�~��������������������������������������-�*�,�-�2�:�F�L�S�_�f�_�\�S�F�:�-�-�-�-�F�D�?�F�K�S�_�l�v�u�l�e�_�S�F�F�F�F�F�F�������}���������������������������������T�:�3�7�B�a�������������������������m�T��¾£¿������/�<�a�o�c�<�#�����DIDADCDIDVDaDbDfDbDVDIDIDIDIDIDIDIDIDIDI���������ùϹչֹϹù�����������������������'�3�@�Y�e�~���������r�L�@�3�-�'������������������ùܹ�������ܹйù����Ľ������Ľнݽ������������ݽнĽ����������������ĽȽνĽ�����������������������������(�3�4�8�4�(�&��D{DuDwD{D{D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{ĿķĳıĹĿ����������� ��������������Ŀ�!���!�'�.�:�G�S�W�S�N�G�:�.�*�!�!�!�!�S�H�L�S�`�d�l�p�l�`�S�S�S�S�S�S�S�S�S�S 5 6 Z * - � m a ?  t Q - 0 ) G H - 0 2 M o * , K ' D n i O U H X = K r G V ' L H I & Z 8 = e - n 3 } H S 5 1 / N D S F 2 � 3 e 6 = A 9 ` >  K  �    �  9  �  &  �  �  q  �  �  D  �  �    I  U  H  �  &  �  �  �  �  �  @    �  �  ]  �  �  �  �  :  :  �  "  �  �  �  '  o        �    �  �    X  F  �  �  �    ,  >  g  �    \  �  �  u  �  �  "  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  g  �  �  �  �  �  b  =    �  �  }  A  �  �  I  �  #  l  �  �  �  �  �  �  �  �  �  �  r  R  0    �  �  �  w  ?   �   �  �  }  l  Y  F  /    �  �  �  `    �  p    �  J  �  r    %  -  0  .  $    �  �  �  �  r  K  #  �  �  I  �  �  P  A  8  *      �  �  �  �  �  �  �  j  L  .    �    4  �  �  M  ?  0      �  �  �  �  l  9    �  t  4  �  �  �  E    �  �  �  ~  t  i  ^  S  H  B  ;  5  ,  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  _  X  P  E  9  -         �  �  �  �  �  �  �  �  �  �  �  (  b  �  �  �    4  <  H  I  6    �  w    u  �  �    �  �  �  �  �  �  x  j  P  4    �  �  �  �  �  �  p  3   �  +  2  8  <  5  /  +  *  )  #      �  �  �  �  �  `  ?    �  �  �  �    	    �  �  �  �  �  �  �  {  c  I  .    �  �  �  �  z  p  g  `  W  F  /    �  �  �  M      �  �  -     X  �  �  �    '  <  L  M  ;    �  �  n    �  �  �   �  �  n  u  �  �  �  �  �  r  =       �  �  �  �  �  Z  "    �  �  �  �    -  D  R  W  Q  A  %  �  �  a    �    w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  d  T  C  i  �  �  �  �      #  !    �  �  `    �  s    �  e    y  k  \  M  ?  0  !      �  �  �  �  �  �  �  �  p  _  N  �  �  �  �  �  �  �  �  �  q  `  Q  C  5  &      �  �  �  R  �  ;  @  @  >  7  -  "    �  �  �  \    �  �  R    �         �  �    ?  t  p  _  G  '    �  �  [  
  �  a    �  �  �  �  {  s  k  c  Z  Q  H  >  6  .  &          �  U  R  O  L  I  F  C  @  <  9  6  1  -  (  $            �  �  z  ^  =    �  �  �  �  �  j  9  �  '  �  �  W  �  �  �  �  �  �  �  �  �  �  r  ]  B  !  �  �    E     �   �   �  �    w  m  [  ?    �  �  o    �  W  '  �  �  A  �  C  w  �  �  �  �  �  c  B  )    �  �  �  h  9    �  �  5  �  X  {  |  {  r  h  \  P  B  4  &      �  �  �  �  �  w  T  1  d  b  Z  F  0    �  �  �  �  z  L  k  r  {  J  	  �    �  -  :  L  W  ]  U  @  $    �  �  �  ~  I  �  A  �    �  �  �  �  �  y  G  8  E    �  �  �  �  �  �  �  M  �  i  �   �  9  M  W  X  P  G  :  *    �  �  �  P  �  �  *  �  *  �  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  T  >  '    �  �  �  �  t  [  9    �  �  �  �  �  �  V  %  �  �  z  9  %  "      3  H  8  (      �  �  �  �  |  Z  9    �  �  �  �  �  �  �  �  x  M    �  �  �  �  r  6  �  �  [  /  4  �  �  �  �  �  �  �  �  �  t  P  '  �  �  W  �  O  �  1    g  �  �  �  {  C    �  	  �  �  k    �  d  �  �  G  �  w    �  �  �  �  �  �  �  �  |  i  R  8    �  �  �  D  �  &  F  7  (        �  �  �  �  �  m  P  2    �  �  �  �  �  i  n  h  \  M  :  #    �  �  �  �  S    �  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  o  S  6    �  �  �  �  �  h  �  �  �  �  �  �  �  r  J  #  �  �  �  N      �  c  �  �  _  U  L  A  4  '      �  �  �  �  �  �  �  �  �  �  �  x  �  �  t  h  a  K  2      �  �  �  �  d  9    �  �  �  �  O  \  h  o  q  o  _  K  2    �  �  �  o  /  �  �    c  O  �  �  �  r  a  M  9  %    �  �  �  �  �  �  �  q  \  H  3  �  ~  r  ^  F  -    �  �  �  �  j  E    �  �  �  k  @  J  �    c  J  ,    �  �  �  �  �  �  �  m  [  J  :  /  ?  c  P  I  A  9  2  )       
  �  �  �  �  �  s  ]  U  N  H  B  O  O  P  Q  L  F  @  +    �  �  �  �  �  �  �  �  �  �  }  ;    �  �  �  �  t  U  6    �  �  �    M    �  �  Q      �  �  �  �  �  �  �  �  �  �  �  t  Z  ?  !    �  �  �  �  �  �  �  �  �  �  �  �  �  t  [  =    �  �  �  �  ^  %  �  �  �  !    �  �  �  `  -  �  �  �  O  
  �  s     �  �  �  ^  9    �  �  K    �  �  �  �  c    �  6  �  )  d   �  �  �  �  �  �  n  @    �  �  �  B  �  �  &  �  �    �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  x  R  Q  a  t  �  �  �  �  �  �  �  �  c  ,  �  |    �  X    �  �  ^  6    �  �  �  k  E    �  �  �  �  |  c  W  >    B  �  �    [  3    �  �  c  #  �  �  0  �  �     �  �  $  D  ?  :  .  !    �  �  �  �  �  �  ~  x  l  \  @     �  �  C  ,    �  �  �  �    _  <    �  �  �  `  *  �  �  �  p  <  U  j  u  k  L  $  �  �  �  ]     �  �  N  �  �    g  �  )    �  �  �  �  �  t  `  K  .  	  �  �  �  �  H  �    Q  �  �  �  �  �  �  g  G  &    �  �  �  s  d  N  2  
  �  �      �  �  �  �  �  �  �  �  �  {  r  i  S  ,    �  �  r                
        
            !  $  (