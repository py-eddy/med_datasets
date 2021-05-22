CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�j~�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�j0   max       P��H        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��"�   max       <T��        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F7
=p��     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v|��
=p     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q@           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �6E�   max       ���
        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�-<   max       B0%        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B0=�        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�j0   max       P�N        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�%F
�L0   max       ?ӕ�$�/        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��"�   max       :�o        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(��   max       @F+��Q�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v{\(�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q@           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�}             U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?ӓݗ�+k        W�   '                           +      '   ,                     !   %   	                              
          #   '   
            [   5         =      	         ,   $      :   ?   &          `         6   #   
   O���N���N�o4O3�O��Ns�XO�p�N8��On,TP9Q�NA��P$��O��OZ�;N9�OS��Nӳ�M�O��O�g�OΑ�Og�tN�bN.dNr�N?w$Ov#jN?�4N[�>O˽O�A�N���O��O!lrO�:�P ��O�FN�%OO�OqPT�+P �YO	N��O�6uN<o@NS��N�{vM�j0P��O���OHP#V�P�POQ��N�N�O(\P��HNZdjOwMOU3O)ZhNi�yNH�<T��:�o%@  �D���D���#�
�#�
�49X�49X�D���D���T���T���u��o���
���
���
���
��j�ě��ě��ě�������/��h�����o�o�\)��P��w�#�
�0 Ž49X�8Q�8Q�D���D���H�9�L�ͽY��Y��q���y�#�y�#�}󶽁%��+��+��\)��hs��hs������-���
���置{��{��^5�����
=��"�)6BI[^a[[ZOC6)}�������������}}}}}}��������������������������������������������*/0)��������fgty����tgeaffffffff����������������������������������������9<GUacnpvvyneaUH=759������
������{wz�
!
	


*/:HT^`edaTH/*6CITX[VOC6*
bgp�������������tgbb
#'&#
")5BN[bhlnqgd[TN5("rt~�����������|txtrr����������������������������������������#/6<EMZVUE5/(#��� )6BNWORNNLB6�<BO[t��������h[OB;;<����������������������������������������������������������������������������������thb^X@=;BO[hlux}��7<=IKLIH><:546777777st���������wtpssssss��������������������RTamz���������zrjdTRFHPTaafjliaTJHHAFFFF�����������������������������������z���������������uuvz#0bn{���{oU<#��������������������SUW_bn{�{x{|{nb]UONS�����������������������##�������$.EUanz������aH8/ $��������������������;<=HUW\inqpnaUOHGD<;���������������������
0ITI;0*#
��������|vtqkimt����������)+3676/)'����������������������������������������/2?HUz��������zgUH5/���������������������������  �����������V_g������������tg]VV���������������������
!#$$&%#"
�����
#,)'%$$#
�������������������������)B[svgUE&���zz����������}zzzzzzz��������������������#/<HOUU`UTHF</#268BFO[^ehkjh][OB962��	������������
	
���������F�5�1�3�:�E�F�S�]�l�x������y�s�l�_�S�F�������������������������������������������������������������������������������������������������
��#�/�3�.�#��
�����Z�N�D�>�C�N�Z�g�������������������s�g�Z�����������������������������������������������g�^�f�i�s�������������������������������������������������������������������������������������	��	�����������˾X�M�F�F�L�^�s�������������Ǿ¾�����f�X�����������ʾ׾����׾Ҿʾ��������������������m�g���������	���	���������������۾оʾȾʾ��	�"�,�5�:�<�7�.�"�	���m�k�a�Y�V�N�K�L�T�a�m�z�|�����������z�m�����������������ʼ̼̼ʼ����������������y�v�v�z�����������ĿѿԿѿ˿Ŀ��������y�ݿտݿ޿�����������������ݿݼ�����������������������������������������������������!�!�-�8�2�-�!���H�<�)� �#�&�/�<�U�a�j�n�zÅ�z�v�n�a�U�H�������z�e�R�L�B�H�T�_�m�z���������������y�s�x�v�w�r�j�e�m�y�������������������y�G�F�;�6�6�;�G�T�`�_�`�b�`�T�G�G�G�G�G�G�ʼɼʼҼּ����ּʼʼʼʼʼʼʼʼʼʻ���������������������������������������������&�'�+�4�9�7�4�'����������������������������������������������׾�����(�3�4�A�D�A�4�(�������������������	�	��	���������𿶿����������������Ŀӿֿۿ�������ѿ�����ƿ����������� �������������̿��ݿؿܿݿ����������������ŭŠŔŇ�n�e�`�e�oŀŔŠŭŹ��������Źŭ�U�a�n�o�x�zÀ�z�s�n�a�[�U�H�:�2�9�<�H�U�r�l�h�m¦´¿��������²¦���������y�n�Q�P�]�s���������������������"��	�������	��"�.�;�G�T�W�\�T�G�;�.�"���ܻлϻλ׻ܻ������������������	�������������������	����$�#�"��	�������������!�-�.�3�2�.�'�!����������m�O�D�A�E�Q�z�������������������޼f�M�B�M�{���������ʼ����ۼʼ������f�������������������ĿѿӿܿԿѿƿĿ����������������Ӻֺ������������ֺɺ��������������������Ľڽ�������ݽнĽ���ƎƎƚƧƫƧƚƎƁ�x�~ƁƎƎƎƎƎƎƎƎìéëìù����������������ùìììììì��������(�5�:�:�5�.�(�����������)�5�8�5�*�)�������������������|�p�q����������þӾ�׾Ѿоʾ���ܻջ̻ʻѻܻ����.�2�)���	������$����$�+�0�=�I�T�V�b�c�b�^�V�I�=�0�$ĚčċĆ�z�tāčĚĦĳĿ������������ĳĚ�����������������
��#�d�n�k�d�_�V�<�#������������������������)�*�%�������D�D�D�D�D�D�D�D�EEEE EEEEED�D�D��ݼܼ��������!�#�"�!������������Ó�n�`�M�1�(�<�H�X�zì����6�E�6����~�}�r�~���������������������~�~�~�~�~�~���������������ɺ޺����	������ɺ���E�E�E�E�E�E�FFFF#F$F.F1F;F:F1F$FFE��ܹҹù��������������ùϹչ������ܽ!� ��!�+�.�9�:�G�R�G�?�:�.�!�!�!�!�!�!�l�k�`�^�`�l�y�����������������y�l�l�l�l   d k =  E D 6 @ [ } O ! b K } c B / ( E � 2 o e Y s Q \ Y � 4 6 # M C t D � D e C b R 3 e J - O 2 - = : 0 I H L � e / & , : �  �  �  �  �  �  �  �  N  �  �  �  0  �    T  �    *        �  �  d  �  W  ^  }  �  (      �  W  �    �  �  �  I  �  �  D  9  �  Y  �  �  #  d    Z  �  p  �    �  }  �  �  �  v  |  Ƽ������
�ě��ě������e`B���e`B��/�aG���C��T���ixռ�`B��1�o��/�������aG��u�+��`B���o�C��P�`��P�t��aG��D���<j��C���hs���w��aG��D�������C��\)��񪽁%��O߽�F��C���\)���罅���;d�����1�o�1'��S��Ƨ��`B�6E���^5��xվz����xս�S�B�8B[�B�cB�sBF�B	��B�B0�B��B!!�B$u�A�-<B0%B
�aB$�vB�~B
�	B 2�B,WBtqB�B�B ��B!�cB�.B h9BZ�B&?�B
o�B+B r�A�ܘB�;B!R^B��B&��B9�B'��B!�CB.J�B�;B+��B�JB!^B$�YB	�#BB@B��B��B�_BzB
��B�B�	B;,B��B�\B��BbBOB�]B��B��B��B�bB�#B�iB��B	�0B3�B:�B�=B 0�B$��A�|�B0=�B�BB$�]B��B
�B ?EB+�B<nB?�B>�B!1PB!�aB�JB c}B�dB&>B
?�B+8�B =�A��B��B!=DBL B&�^B<YB'�7B!I�B.}B��B+�B:�B ��B$��B	�yB �B?RB�(B��B�"B�B
XXB<UB��B=tB�JBC�B�RBL�B>�B��B�1B�u@���A�_cA��\A�[�A���A��aA���A�?�A�%�AEy�AP�+A�dAZ9�A�/�@��GAs��A�+@�H}@`C�A��A��Am�AfqA �n@��M@ǜkA��A6�CAX�.Azy�B` A��A�*VA�΍A�YAA�q�A`;+@���A�JRA
tA��@�O$Av��@:�.A&HB�A���A���A�[$AJ�Q@���B
��A���A�"Aһ�C�@/A��A͋U@�@<b�C���>���A�A �@��Aэ�A�OBA��cA��A���A�όA�cDA��AB�kAR��A�pAZ��A��@��{Aq'A�u9@�1�@d�A�[�A�a�Am�Ae��A �@��@��wA���A7�AY �AxMBw~A��A�BA��A��A�tPAao@���A��A
�:A���@�S�Ay�@8E!A% B�2A͋A���A��AH��@��BB
��A��GA�TAҀ�C�8�A�]A��{@�@C�C���>���A!2AW�   (                           ,      (   -                     !   %   
                              
          $   '               \   6         =      
         ,   $      :   ?   '      !   `         7   $   
                  !      !         /      )   !                        $                           )   #      !      !   +               ;   )         !               '         +   %            G                                 !               )                                 $                           )   #            !   +               #   )                        #                        A                  O�6N���Nj��O3�O��Ns�XO���N8��O*\O��2NA��O�O[��OZ�;N9�O:��N��5M�O��ONA�OΑ�Og�tN�bN.dNr�N?w$Ov#jN?�4N[�>O˽O�A�N���O�
�O!lrO�:�P ��O�FN�%OO�OqO�!QP �YO	N��O���N<o@NS��N�{vM�j0O�HtOW{�OHO�v�O��O7��N��sO(\P�NNZdjOwMN�!}N��qNi�yNH�  �  �  R  �  D  �  4  9  �  �  Y  6  $  �  
  �  /  ^  �    �      j  |    �    v  �  �  �    �  t  �  �  �  �  U  	�  ~  ;  �  	I    �  �  �  �  z    �  �  K  P  	�  �  )  *  
�    �  
��o:�o��o�D�����
�#�
�e`B�49X�e`B��9X�D�����ͼ�/�u��o��1��1���
���
��/�ě��ě��ě�������/��h�����o�o�\)��P�49X�#�
�0 Ž49X�8Q�8Q�D���D�����P�L�ͽY��Y���7L�y�#�y�#�}󶽁%��O߽�hs��\)��9X������㽣�
���
��E���{��{�����
=��
=��"�')6BIORRQOHB:6)&!"''}�������������}}}}}}���������������������������������������������'-.)������fgty����tgeaffffffff����������������������������������������9<>HNU^afnrqnjaUH@;9}���������������}}
!
	


"'/;HTX\^\ZTH;4"!"'*6CDKORROHC6*bgp�������������tgbb
#'&#
&)5BN[agjkh[VN@85*"&{�����������~v{{{{{{����������������������������������������#/<@HPUURH</#��� )6BNWORNNLB6�<BO[t��������h[OB;;<����������������������������������������������������������������������������������thb^X@=;BO[hlux}��7<=IKLIH><:546777777st���������wtpssssss��������������������RTamz���������zrjdTRFHPTaafjliaTJHHAFFFF���������� �����������������������������z���������������uuvz#0bn{���{oU<#��������������������SUW_bn{�{x{|{nb]UONS�����������������������##�������JMUanz�������nUHEABJ��������������������;<=HUW\inqpnaUOHGD<;���������������������
#0<=:6+#
��������|vtqkimt����������)+3676/)'����������������������������������������<AHUz�������znUH;24<���������������������������  �����������agu������������tf^^a���������������������
###%#! 
 �����
#)'%# 
 ���������������������4N[giNC$������zz����������}zzzzzzz��������������������#/:<GHIHF<;/,# ?BJO[_ge^[OB><??????��	������������
	
���������F�C�=�C�F�L�S�_�l�q�x�{�x�w�n�l�_�S�F�F�������������������������������������������������������������������������������������������������
��#�/�3�.�#��
�����g�Z�N�G�A�B�G�O�Z�g�����������������s�g�����������������������������������������g�m�s������������������������������s�g��������������������������������������������������������������������� � �������׾s�f�Z�M�L�P�\�m�s�������������������s�����������ʾ׾����׾Ҿʾ������������������������������������� ������������������־վ׾����	��"�'�.�/�+�"����m�k�a�Y�V�N�K�L�T�a�m�z�|�����������z�m�����������������ʼ̼̼ʼ����������������y�x�w�{�����������ĿѿɿĿ������������y�����������������������꼋����������������������������������������������������!�!�-�8�2�-�!���H�E�<�,�$�+�/�;�H�T�U�a�i�n�x�o�a�U�K�H�������z�e�R�L�B�H�T�_�m�z���������������y�s�x�v�w�r�j�e�m�y�������������������y�G�F�;�6�6�;�G�T�`�_�`�b�`�T�G�G�G�G�G�G�ʼɼʼҼּ����ּʼʼʼʼʼʼʼʼʼʻ���������������������������������������������&�'�+�4�9�7�4�'����������������������������������������������׾�����(�3�4�A�D�A�4�(�������������������	�	��	���������𿶿����������������Ŀӿֿۿ�������ѿ�����ƿ����������� �������������̿��ݿؿܿݿ����������������ŔōŇ�{�n�i�k�uŇŔŠŹŽ������ŹŭŠŔ�U�a�n�o�x�zÀ�z�s�n�a�[�U�H�:�2�9�<�H�U�r�l�h�m¦´¿��������²¦���������y�n�Q�P�]�s���������������������"��	�������	��"�.�;�G�T�W�\�T�G�;�.�"���ܻлϻλ׻ܻ������������������	�������������������	����$�#�"��	�������������!�-�.�3�2�.�'�!������z�m�^�Q�M�R�_�m�z���������������������z�f�M�B�M�{���������ʼ����ۼʼ������f�������������������ĿѿӿܿԿѿƿĿ����������������Ӻֺ������������ֺɺ��������������������Ľݽ����ݽнĽ�����ƎƎƚƧƫƧƚƎƁ�x�~ƁƎƎƎƎƎƎƎƎìéëìù����������������ùìììììì��������(�5�:�:�5�.�(�����������)�5�8�5�*�)�����������������}�q�s�����������̾Ծվξξʾ���������ڻлλлֻܻ������'�#�������$����$�+�0�=�I�T�V�b�c�b�^�V�I�=�0�$ĚĔčċĆĀ�}āčĦĺĿ����������ĳĦĚ�#������
��#�0�:�I�Q�U�R�K�I�<�0�#����������������������(�#��������D�D�D�D�D�D�D�D�EEE	E	ED�D�D�D�D�D�D��ݼܼ��������!�#�"�!���������n�b�S�H�/�.�<�HÇì����=�6�����àÇ�n�~�}�r�~���������������������~�~�~�~�~�~���������������ɺ޺����	������ɺ���E�E�E�E�FFFFFF$F0F1F3F1F*F$FFE�E��ù����������ùϹܹ��߹ܹϹùùùùùý!� ��!�+�.�9�:�G�R�G�?�:�.�!�!�!�!�!�!�l�k�`�^�`�l�y�����������������y�l�l�l�l  d b =  E > 6 # U } F % b K x N B / % E � 2 o e Y s Q \ Y � 4 / # M C t D � D 2 C b R # e J - O 5 # = :  G 1 L � e /   $ : �    �  �  �  �  �  %  N  p  �  �  �  �    T    �  *    �    �  �  d  �  W  ^  }  �  (         W  �    �  �  �  I    �  D  9  :  Y  �  �  #  0  �  Z  �  �  �  �  �  (  �  �    �  |  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  e    �  �  �  �  �  �  �  �  �  �  ^  &  �  �  8  �  �  [  �  �  �  �  {  v  p  g  \  Q  C  1       �  �  �  �  �  �  .  9  D  P  O  L  J  >  -      �  �  �  �  �  �  [  *   �  �  �  �  �  �  �  �  �  ]  4    �  �  �  `  %  �  4  �  Z  :  A  A  6  '        �  �  �  �  �  r  D    �  �  A   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  v  p  k  �    1  4  0  (        
      &  )    �  �  G  �  �  9  1  *  "        �  �  �  �  �  �  �  s  W  ;        �  i  h  f  �  �  �  �  �  p  \  F  /    �  �  �  s  >  �  V  �  �  �  �  �  �  �  �  �  �  p  C    �  a  &  �  �  �   �  Y  [  ]  `  ^  M  =  ,  .  N  m  �  k    �  �    �  ,  �  �  �      +  3  6  3  /  %    �  �  �  W  �  U  �     �  �  �  �         $       �  �  �    Q    �  u    �  �  �  �  �  ~  k  Z  K  <  5  1  ,  '      �  �  �  �  l  ^  
  �  �  �  �  �  �  �  �  �  p  \  E  ,    �  �  �  �  �  �  �  �  �  �  �  t  W  8    �  �  �  p  G  �  �  S   �   {  
    %  .  *  '         �  �  �  �  �  �  p  D     �   �  ^  ^  ^  ^  \  Z  X  N  B  5  '      �  �  �  �  �  �  q  �  �  �  �  �  y  e  P  >  -        �  �  �  �  �  �  �  �          	  �  �  �  �  �    ^  1  �  �  |  ?  /  @  �  �  �  �  �  }  p    �  j  4  �  �  1  �  0  �    �   �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  ]      
        �  �  �  �  �  �  �  �  ~  d  G  *     �  j  d  ^  Y  S  L  B  7  -  "      �  �  �  �  �  �  �  �  |  s  k  c  Z  S  K  C  4      �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  I  $    .    �  �  �  �  `  %  �  �    �      �  �  �  �  �  �  �  �  �  v  e  S  =  '    �  �  �  v  o  g  `  Y  Q  H  @  7  .      �  �  �  �  i  <     �  �  �  �  �  �  �  u  ^  F  -    �  �  �  {  L    �  �  E  �    ]  ;    �  �  �  �  �    �  �  �  �  �  a  4    �  �  �  �  �  �  �  �  �  �  �  �  �  r  [  ;    �  �  �  �    
        	    �  �  �  �  �  Z  $  �  �  u  %  �  '  �  �  �  y  M    �  �  g  (  �  �  b    �  b  �  �  �   �  t  _  7    �  �  I    �  �  �  �  �  �  z  R     �  �  v  �  c  A    �  �  �  �  �  �  �  b  7    �  �  �  6  �    �  ~  m  b  W  L  ?  .    �  �  �  h  C  4  /    �  �  E  �  �  �  �  �  �  �  �  �  �    |  z  s  f  X  J  =  /  !  �  �  �  �  y  s  i  I    �  �  |  M  B  2  �  �  4  �  �  U  >  &      �  �  �  �  ^  6    �  �  �  N    �  s    �  �  	+  	a  	t  	�  	�  	|  	n  	R  	+  �  �  M  �  S  �  �  �  R  ~  y  n  Y  F  +    �  �  �  d  9  	  �  �    �  
  8   �  ;    �  �  �  �  x  R  '  �  �  �  �  �  ^  &  �  �  O    �  �  �  �  �  h  N  1    �  �  �  �  i  O  A  h  �  �  �  {  	  	6  	G  	>  	.  	  �  �  v  '  �  d  �  Z  �    M  ]      �  �  �  v  Z  ?  #    �  �  �  �  t  W  ;      �  �  �  �  �  �  �  �  s  V  7    �  �  �  z  [  @  2  $      �  �  �  �  �  s  U  1    �  �  �  N    �    t  �  #   q  �  �  �  �  �  �  �  �  �  �  s  f  Y  L  ?  2  %       �  �  �  �  �  �  �  �  {  T    �  �  ,  �  6  �  �  L    v  i  v  y  y  q  _  @    �  �  ~  ?  �  �  [  �  �  �  e  �    �  �  �  �  P    �  �  K    �  �  f  "  �  �  ;   �   t  �  �  ?  |  �  �  �  �  �  `    �  l  �  `  �  �  6  �  3  �     p  �  �  �  �  �  �  �  �  �  D  �  {  �  _  z  b  0  @  I  G  >  "  �  �  �  i  ,  �  �  r  3  �  �  3  �  z  E  �    E  O  K  B  )  	  �  �    6  �  �  ?  �  !  {  �  !  	�  	e  	<  	
  �  �  S    �  ~  2  �  �  -  �  '  �    !  M  �  �  �  �  p      
�  
a  	�  	�  	  �  r  4  �  �  t  �  O  )              �  �  �  �  u  U  5    �  �  �  �  \  *    �  �  �  �  t  N    �  �  k  1  �  �  �  U    �    
%  
e  
�  
�  
�  
�  
�  
�  
z  
`  
6  
  	�  	[  �  ~  �  �  *  �  �  �  �  �  �  
          �  �  y  8  �  �  Z    �  :  �  �  �  s  E    �  �  �  �  �  �  �  �  �  �  |  x  v  t  
  �  �  �  �  �  �  q  V  ;  !    �  �  �  �  �  t  \  D